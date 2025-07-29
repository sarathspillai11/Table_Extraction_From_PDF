import os
import json
import base64
import traceback
from openai import AzureOpenAI
from typing import Dict, List, Any

class TableImageAnalyzer:
    def __init__(self, azure_endpoint: str, api_key: str, api_version: str = "2024-02-01", deployment_name: str = "gpt-4-vision"):
        """
        Initialize the Azure OpenAI client for image analysis
        
        Args:
            azure_endpoint: Your Azure OpenAI endpoint URL
            api_key: Your Azure OpenAI API key
            api_version: API version (default: "2024-02-01")
            deployment_name: Name of your GPT-4 Vision deployment
        """
        self.client = AzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version=api_version
        )
        self.deployment_name = deployment_name
    
    def encode_image(self, image_path: str) -> str:
        """
        Encode image to base64 string
        
        Args:
            image_path: Path to the PNG image file
            
        Returns:
            Base64 encoded image string
        """
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except FileNotFoundError:
            print(f"Error: Image file not found at path: {image_path}")
            print("Full stacktrace:")
            traceback.print_exc()
            raise
        except PermissionError:
            print(f"Error: Permission denied when trying to read file: {image_path}")
            print("Full stacktrace:")
            traceback.print_exc()
            raise
        except Exception as e:
            print(f"Error encoding image: {e}")
            print("Full stacktrace:")
            traceback.print_exc()
            raise
    
    def analyze_table_image(self, image_path: str) -> Dict[str, Any]:
        """
        Analyze table image and extract structured data
        
        Args:
            image_path: Path to the PNG image containing the table
            
        Returns:
            Dictionary containing the analysis results
        """
        base64_image = self.encode_image(image_path)
        
        prompt = """
        Analyze this table image and extract the following information in JSON format:

        1. Identify all row headers and column headers
        2. For each cell in the table, provide:
           - row_header: The header for that row
           - column_header: The header for that column
           - cell_value: The actual value in the cell
           - search_question: A natural language question that could be asked to a document search system to retrieve this specific value

        Format the output as a JSON object with this structure:
        {
            "table_metadata": {
                "row_headers": ["header1", "header2", ...],
                "column_headers": ["header1", "header2", ...]
            },
            "cells": [
                {
                    "row_header": "string",
                    "column_header": "string", 
                    "cell_value": "string",
                    "search_question": "string"
                }
            ]
        }

        For the search_question, create questions that would help retrieve the specific data point from a document database. For example:
        - If the cell shows "Q3 2023 Revenue: $1.2M", the question might be "What was the revenue in Q3 2023?"
        - If the cell shows "Employee Count: 150", the question might be "How many employees were there in [time period]?"

        Make sure to capture every cell in the table and provide meaningful search questions for each.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=4000,
                temperature=0.1
            )
            
            # Extract the JSON from the response
            content = response.choices[0].message.content
            
            # Try to extract JSON from the response
            # Sometimes the model wraps the JSON in code blocks
            if "```json" in content:
                json_start = content.find("```json") + 7
                json_end = content.find("```", json_start)
                json_content = content[json_start:json_end].strip()
            elif "```" in content:
                json_start = content.find("```") + 3
                json_end = content.find("```", json_start)
                json_content = content[json_start:json_end].strip()
            else:
                json_content = content.strip()
            
            # Parse the JSON
            result = json.loads(json_content)
            return result
            
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON response: {e}")
            print(f"Raw response: {content}")
            print("Full stacktrace:")
            traceback.print_exc()
            return {"error": "Failed to parse JSON response", "raw_response": content}
        except Exception as e:
            print(f"Error analyzing image: {e}")
            print("Full stacktrace:")
            traceback.print_exc()
            return {"error": str(e)}
    
    def save_to_json(self, data: Dict[str, Any], output_path: str):
        """
        Save the analysis results to a JSON file
        
        Args:
            data: The analysis results dictionary
            output_path: Path where to save the JSON file
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"Results saved to: {output_path}")
        except PermissionError:
            print(f"Error: Permission denied when trying to write to: {output_path}")
            print("Full stacktrace:")
            traceback.print_exc()
            raise
        except json.JSONEncodeError as e:
            print(f"Error encoding data to JSON: {e}")
            print("Full stacktrace:")
            traceback.print_exc()
            raise
        except Exception as e:
            print(f"Error saving JSON file: {e}")
            print("Full stacktrace:")
            traceback.print_exc()
            raise

def main():
    # Configuration - Replace with your Azure OpenAI details
    AZURE_ENDPOINT = "https://your-resource-name.openai.azure.com/"
    API_KEY = "your-api-key-here"
    API_VERSION = "2024-02-01"
    DEPLOYMENT_NAME = "gpt-4-vision"  # Replace with your deployment name
    
    # You can also use environment variables for security
    # AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    # API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
    # DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    
    # File paths
    image_path = "table_image.png"  # Replace with your image path
    output_path = "table_analysis.json"
    
    try:
        # Validate inputs
        if not os.path.exists(image_path):
            print(f"Error: Image file not found at {image_path}")
            return
        
        if not AZURE_ENDPOINT or not API_KEY:
            print("Error: Please configure your Azure OpenAI credentials")
            return
        
        # Initialize analyzer
        print("Initializing Azure OpenAI analyzer...")
        analyzer = TableImageAnalyzer(
            azure_endpoint=AZURE_ENDPOINT,
            api_key=API_KEY,
            api_version=API_VERSION,
            deployment_name=DEPLOYMENT_NAME
        )
        
        # Analyze the image
        print(f"Analyzing image: {image_path}")
        results = analyzer.analyze_table_image(image_path)
        
        # Check for errors
        if "error" in results:
            print(f"Analysis failed: {results['error']}")
            if "raw_response" in results:
                print("Raw response from API:")
                print(results["raw_response"])
            return
        
        # Save results
        print("Saving results to JSON file...")
        analyzer.save_to_json(results, output_path)
        
        # Print summary
        if "table_metadata" in results:
            print(f"\nAnalysis Summary:")
            print(f"Row headers: {len(results['table_metadata']['row_headers'])}")
            print(f"Column headers: {len(results['table_metadata']['column_headers'])}")
            print(f"Total cells analyzed: {len(results['cells'])}")
            
            # Print first few cells as examples
            print(f"\nFirst 3 cells (example):")
            for i, cell in enumerate(results['cells'][:3]):
                print(f"  {i+1}. [{cell['row_header']}] x [{cell['column_header']}] = '{cell['cell_value']}'")
                print(f"     Search question: {cell['search_question']}")
    
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
    except Exception as e:
        print(f"Unexpected error in main function: {e}")
        print("Full stacktrace:")
        traceback.print_exc()

if __name__ == "__main__":
    main()
