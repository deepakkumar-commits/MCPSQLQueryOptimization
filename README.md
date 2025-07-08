
# SQL Query Optimization Architecture with Model Context Protocol 

## SQL Query Optimization with MCP Basic Example
This is a basic example of how  Model Context Protocol (MCP) can be used for SQL query optimization. It demonstrates core functionality including tools and resources.

## MCP Architecture for SQL Query Optimization

## Server-Side Components
<img width="452" alt="image" src="https://github.com/user-attachments/assets/f3e76a0b-e8b2-42e0-a166-34ca868ec994" />


The MCP server architecture consists of several key components working in harmony:

## Core Framework
-	FastMCP Server: Handles protocol communication and request routing
-	Database Connection Class: Manages database connectivity across different engines
-	Dialect Support: Adapts to various SQL dialects (PostgreSQL, MySQL, SQL Server, etc.)

## MCP Tools Suite
The optimization workflow encompasses seven distinct tools: 
- Table Extractor: Identifies tables and relationships from SQL queries
- Complexity Analyzer: Evaluates query complexity and resource requirements 
- Execution Plan Generator: Creates and analyzes query execution plans 
- Index Advisor: Recommends optimal indexing strategies 
-	Join Optimizer: Suggests join reordering and optimization 
-	Performance Predictor: Estimates query performance metrics 
-	Report Consolidator: Generates comprehensive optimization recommendations

## External Dependencies
- Language Model Integration: Connects with OpenAI, Anthropic, and other LLM providers
-	Database Connectors: Supports multiple database engines and cloud platforms

## Client-Side Components

The MCP client orchestrates the optimization workflow through:

## Core Components
- Connection Management: Handles MCP server communication
- Async Workflow Adapter: Manages concurrent optimization tasks
- Input Processing: Accepts queries from multiple sources (stdin, files, interactive mode)

## Workflow Execution Engine
- State Management: Tracks optimization progress and intermediate results
- Async Execution: Enables parallel processing of optimization steps
- Error Handling: Provides robust error recovery and retry mechanisms

## Report Generation
- Multi-format Output: Generates reports in text, HTML, JSON, and PDF formats
- Visualization: Creates performance charts and optimization comparisons
- Export Capabilities: Integrates with popular reporting tools

## Data Flow and Processing Pipeline

The MCP-enabled optimization process follows a structured seven-step workflow:

<img width="321" alt="image" src="https://github.com/user-attachments/assets/446ef8a0-23d6-4c15-a2a9-6bf4b857ab97" />

## Setup Steps
Initialize the project and create the virtual environment:
- uv init mcp_sqlquery_demo
- cd mcp_sqlquery_demo
- uv venv
- .venv\Scripts\activate

Install all the dependencies given in the requirement.txt and set the environment as given in the environment_template file

## Running the Server and client code
To run the server with the MCP Inspector for development:

 - uv run query_optimizer_mcp_server.py
 - uv run query_optimizer_mcp_client.py

## Optimization report is generated in the required format

<img width="320" alt="image" src="https://github.com/user-attachments/assets/1805edb9-0334-4451-999b-b446fc013fec" />

## Optimization Report generated for a Sql query

 

![image](https://github.com/user-attachments/assets/3b9d6045-eea8-4182-b2f5-c5f750299585)


## Tool Integration and Configuration

Cursor IDE Integration
MCP tools can be seamlessly integrated into popular development environments:

 <img width="320" alt="image" src="https://github.com/user-attachments/assets/af2a7494-5151-4853-aa98-4c2e76be1a3d" />

 
 
<img width="447" alt="image" src="https://github.com/user-attachments/assets/964e27b6-54da-4c7b-bbcd-2b2a94c7348e" />


 


![image](https://github.com/user-attachments/assets/f262fae1-0224-4769-8c6f-4914f0934dd7)

