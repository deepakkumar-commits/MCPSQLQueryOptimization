#!/usr/bin/env python3
# SQL Query Optimizer - Client Implementation (LangGraph Version) - FIXED

import os
import re
import sys
import json
import enum
import logging
import asyncio
from contextlib import asynccontextmanager

from typing import Optional, Dict, Any, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("query_optimizer_client")

# Import MCP related modules
try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
except ImportError:
    logger.error("MCP package is required for this script")
    logger.error("Install with: pip install mcp")
    sys.exit(1)

# Import LangGraph related modules
try:
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint.memory import MemorySaver
except ImportError:
    logger.error("LangGraph package is required for this script")
    logger.error("Install with: pip install langgraph")
    sys.exit(1)

# Try to load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    logger.warning("python-dotenv not installed, not loading .env file")
    logger.warning("Install with: pip install python-dotenv")

# Define database dialects
class DatabaseDialect(str, enum.Enum):
    """Supported database dialects"""
    MYSQL = "mysql"
    POSTGRESQL = "postgresql"
    SQLITE = "sqlite"

# Sample SQL query for testing and demonstration
SAMPLE_QUERY = """
SELECT c.customer_name, o.order_date, p.product_name, p.price
    FROM customers c
    JOIN orders o ON c.customer_id = o.customer_id
    JOIN order_items oi ON o.order_id = oi.order_id
    JOIN products p ON oi.product_id = p.product_id
    WHERE o.order_date > '2023-01-01'
"""

# Configuration from environment variables
CONFIG = {
    # Server configuration
    "SERVER_PATH": os.environ.get("SQL_OPTIMIZER_SERVER_PATH", "sql_query_optimizer_server.py"),
    "SERVER_HOST": os.environ.get("SQL_OPTIMIZER_SERVER_HOST", None),
    "SERVER_PORT": int(os.environ.get("SQL_OPTIMIZER_SERVER_PORT", "0") or 0) or None,
    
    # Database configuration
    "DB_DIALECT": os.environ.get("SQL_OPTIMIZER_DB_DIALECT", "mysql"),
    "DB_HOST": os.environ.get("SQL_OPTIMIZER_DB_HOST", None),
    "DB_USER": os.environ.get("SQL_OPTIMIZER_DB_USER", None),
    "DB_PASSWORD": os.environ.get("SQL_OPTIMIZER_DB_PASSWORD", None),
    "DB_NAME": os.environ.get("SQL_OPTIMIZER_DB_NAME", None),
    
    # Output configuration
    "OUTPUT_FORMAT": os.environ.get("SQL_OPTIMIZER_OUTPUT_FORMAT", "text"),
    "OUTPUT_FILE": os.environ.get("SQL_OPTIMIZER_OUTPUT_FILE", None),
    
    # Visualization
    "GENERATE_VISUALIZATION": os.environ.get("SQL_OPTIMIZER_GENERATE_VISUALIZATION", "false").lower() == "true",
    "VISUALIZATION_PATH": os.environ.get("SQL_OPTIMIZER_VISUALIZATION_PATH", "workflow_graph.html"),
    
    # Logging level
    "LOG_LEVEL": os.environ.get("SQL_OPTIMIZER_LOG_LEVEL", "INFO"),
    
    # Use sample query flag
    "USE_SAMPLE_QUERY": os.environ.get("SQL_OPTIMIZER_USE_SAMPLE_QUERY", "true").lower() == "true",
}

# Configure logging level
logging.getLogger().setLevel(getattr(logging, CONFIG["LOG_LEVEL"]))

class MCPClientManager:
    """
    Manager for MCP client sessions with proper async context management
    """
    
    def __init__(self, server_path=None, host=None, port=None):
        """
        Initialize the MCP client manager
        
        Args:
            server_path: Path to server script (for stdio mode)
            host: Server host (for TCP mode)
            port: Server port (for TCP mode)
        """
        self.server_path = server_path
        self.host = host
        self.port = port
        self.session = None
        self.tools = {}
        self._stdio_context = None
        self._session_context = None
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
    
    async def initialize(self):
        """Initialize connection to MCP server"""
        if self.host and self.port:
            # TODO: Implement TCP client connection
            logger.error("TCP client not implemented yet")
            raise NotImplementedError("TCP client not implemented yet")
        elif self.server_path:
            # Use stdio client with proper context management
            server_params = StdioServerParameters(
                command="python",
                args=[self.server_path],
                env=None
            )
            
            # Create and store stdio client context
            self._stdio_context = stdio_client(server_params)
            read, write = await self._stdio_context.__aenter__()
            
            # Create and store session context
            self._session_context = ClientSession(read, write)
            self.session = await self._session_context.__aenter__()
        else:
            raise ValueError("Either server_path or host/port must be provided")
        
        # Initialize session
        await self.session.initialize()
        
        # Get available tools
        tools_response = await self.session.list_tools()
        self.tools = {tool.name: tool for tool in tools_response.tools}
        
        logger.info(f"Connected to MCP server with {len(self.tools)} tools available")
        return self
    
    async def close(self):
        """Clean up resources properly"""
        try:
            # Close session context
            if self._session_context and self.session:
                await self._session_context.__aexit__(None, None, None)
                self.session = None
                self._session_context = None
            
            # Close stdio context
            if self._stdio_context:
                await self._stdio_context.__aexit__(None, None, None)
                self._stdio_context = None
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    async def call_tool(self, tool_name, arguments):
        """
        Call an MCP tool
        
        Args:
            tool_name: Name of the tool to call
            arguments: Arguments to pass to the tool
            
        Returns:
            Tool response or error dictionary
        """
        if not self.session:
            raise RuntimeError("MCP session not initialized")
        
        if tool_name not in self.tools:
            raise ValueError(f"Tool {tool_name} not available")
        
        try:
            # Call the tool
            result = await self.session.call_tool(tool_name, arguments)
            
            # Process response
            if result.content:
                for content in result.content:
                    if hasattr(content, 'text'):
                        # Parse JSON response
                        try:
                            return json.loads(content.text)
                        except:
                            return {"error": f"Invalid JSON response from {tool_name}"}
            
            return {"error": "Empty response"}
            
        except Exception as e:
            logger.error(f"Error calling {tool_name}: {str(e)}")
            return {"error": f"Tool call failed: {str(e)}"}

class AsyncLangGraphMCPAdapter:
    """
    Async Adapter to integrate MCP tools with LangGraph - keeps everything async
    """
    
    def __init__(self, client_manager, db_connection_params=None):
        """
        Initialize the adapter
        
        Args:
            client_manager: MCP client manager
            db_connection_params: Optional database connection parameters
        """
        self.client = client_manager
        self.db_connection_params = db_connection_params
    
    async def extract_tables(self, state):
        """Call extract_tables MCP tool"""
        result = await self.client.call_tool("extract_tables", {"state": state})
        # Update state with result and return
        new_state = state.copy()
        new_state.update(result)
        return new_state
    
    async def analyze_complexity(self, state):
        """Call analyze_complexity MCP tool"""
        result = await self.client.call_tool("analyze_complexity", {"state": state})
        new_state = state.copy()
        new_state.update(result)
        return new_state
    
    async def get_explain_plan(self, state):
        """Call get_explain_plan MCP tool"""
        arguments = {"state": state}
        if self.db_connection_params:
            arguments["db_connection_params"] = self.db_connection_params
        
        result = await self.client.call_tool("get_explain_plan", arguments)
        new_state = state.copy()
        new_state.update(result)
        return new_state
    
    async def analyze_explain_plan(self, state):
        """Call analyze_explain_plan MCP tool"""
        result = await self.client.call_tool("analyze_explain_plan", {"state": state})
        new_state = state.copy()
        new_state.update(result)
        return new_state
    
    async def recommend_indexes(self, state):
        """Call recommend_indexes MCP tool"""
        arguments = {"state": state}
        if self.db_connection_params:
            arguments["db_connection_params"] = self.db_connection_params
        
        result = await self.client.call_tool("recommend_indexes", arguments)
        new_state = state.copy()
        new_state.update(result)
        return new_state
    
    async def optimize_query(self, state):
        """Call optimize_query MCP tool"""
        result = await self.client.call_tool("optimize_query", {"state": state})
        new_state = state.copy()
        new_state.update(result)
        return new_state
    
    async def generate_final_recommendations(self, state):
        """Call generate_final_recommendations MCP tool"""
        result = await self.client.call_tool("generate_final_recommendations", {"state": state})
        new_state = state.copy()
        new_state.update(result)
        return new_state

def router(state):
    """
    Router function for LangGraph to determine next step
    
    Args:
        state: Current state
        
    Returns:
        Next step or END
    """
    # Check for errors - terminate if critical error
    if state.get("error") and "critical" in state["error"].lower():
        return "complete"
    
    # Get the next step from the state
    next_step = state.get("next_step", "extract_tables")
    
    # Handle completion
    if next_step == "complete":
        return END
    
    # Return the next step
    return next_step

class QueryOptimizerWithLangGraph:
    """
    Query Optimizer using LangGraph for workflow orchestration and MCP for tools
    """
    
    def __init__(self, server_path=None, host=None, port=None):
        """
        Initialize the query optimizer
        
        Args:
            server_path: Path to server script (for stdio mode)
            host: Server host (for TCP mode)
            port: Server port (for TCP mode)
        """
        self.server_path = server_path
        self.host = host
        self.port = port
        self.client_manager = None
        self.adapter = None
        self.workflow = None
        self.db_connection_params = None
    
    async def __aenter__(self):
        """Async context manager entry"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
    
    async def initialize(self, db_connection_params=None):
        """
        Initialize the optimizer
        
        Args:
            db_connection_params: Optional database connection parameters
        """
        # Initialize MCP client with proper context management
        self.client_manager = MCPClientManager(
            self.server_path, self.host, self.port
        )
        await self.client_manager.initialize()
        
        # Initialize adapter
        self.adapter = AsyncLangGraphMCPAdapter(self.client_manager, db_connection_params)
        
        # Store database connection parameters
        self.db_connection_params = db_connection_params
        
        # Create workflow - but don't compile it with LangGraph due to sync/async issues
        # Instead, we'll implement our own async workflow
        
        return self
    
    async def close(self):
        """Clean up resources"""
        if self.client_manager:
            await self.client_manager.close()
    
    async def optimize_query(self, query, dialect=DatabaseDialect.MYSQL):
        """
        Optimize a SQL query using manual async workflow instead of LangGraph
        
        Args:
            query: SQL query to optimize
            dialect: Database dialect
            
        Returns:
            Optimization results
        """
        if not self.adapter:
            await self.initialize()
        
        # Initialize the state
        state = {
            "query": query,
            "dialect": dialect,
            "optimization_steps": [],
            "performance_metrics": {},
            "current_analysis": "",
            "explain_plan": None,
            "final_optimized_query": None,
            "final_recommendations": None,
            "index_recommendations": [],
            "error": None,
            "complexity_score": None,
            "tables": [],
            "next_step": "extract_tables"
        }
        
        logger.info(f"Starting query optimization for query: {query[:100]}...")
        
        try:
            # Execute the workflow steps manually in async context
            # Step 1: Extract tables
            if state.get("next_step") == "extract_tables":
                logger.info("Step 1: Extracting tables...")
                state = await self.adapter.extract_tables(state)
                state["optimization_steps"].append("Extracted tables from query")
            
            # Step 2: Analyze complexity
            if state.get("next_step") == "analyze_complexity":
                logger.info("Step 2: Analyzing complexity...")
                state = await self.adapter.analyze_complexity(state)
                state["optimization_steps"].append("Analyzed query complexity")
            
            # Step 3: Get explain plan
            if state.get("next_step") == "get_explain_plan":
                logger.info("Step 3: Getting explain plan...")
                state = await self.adapter.get_explain_plan(state)
                state["optimization_steps"].append("Retrieved execution plan")
            
            # Step 4: Analyze explain plan
            if state.get("next_step") == "analyze_explain_plan":
                logger.info("Step 4: Analyzing explain plan...")
                state = await self.adapter.analyze_explain_plan(state)
                state["optimization_steps"].append("Analyzed execution plan")
            
            # Step 5: Recommend indexes
            if state.get("next_step") == "recommend_indexes":
                logger.info("Step 5: Recommending indexes...")
                state = await self.adapter.recommend_indexes(state)
                state["optimization_steps"].append("Generated index recommendations")
            
            # Step 6: Optimize query
            if state.get("next_step") == "optimize_query":
                logger.info("Step 6: Optimizing query...")
                state = await self.adapter.optimize_query(state)
                state["optimization_steps"].append("Generated optimized query")
            
            # Step 7: Generate final recommendations
            if state.get("next_step") == "generate_final_recommendations":
                logger.info("Step 7: Generating final recommendations...")
                state = await self.adapter.generate_final_recommendations(state)
                state["optimization_steps"].append("Generated final recommendations")
            
            # Return the optimization results
            return {
                "original_query": query,
                "optimized_query": state.get("final_optimized_query"),
                "analysis": state.get("current_analysis"),
                "final_recommendations": state.get("final_recommendations"),
                "index_recommendations": state.get("index_recommendations", []),
                "optimization_steps": state.get("optimization_steps", []),
                "complexity_score": state.get("complexity_score"),
                "explain_plan": state.get("explain_plan"),
                "error": state.get("error")
            }
        except Exception as e:
            logger.error(f"Error in optimization workflow: {str(e)}")
            return {
                "original_query": query,
                "error": f"Query optimization failed: {str(e)}"
            }
    
    def generate_report(self, results, format="text"):
        """
        Generate optimization report
        
        Args:
            results: Optimization results
            format: Report format (text, html, json)
            
        Returns:
            Formatted report
        """
        if format == "json":
            return json.dumps(results, indent=2)
        elif format == "html":
            return self._generate_html_report(results)
        else:
            return self._generate_text_report(results)
    
    def _generate_text_report(self, results):
        """Generate text report"""
        report = []
        report.append("="*80)
        report.append("SQL QUERY OPTIMIZATION REPORT")
        report.append("="*80)
        
        report.append("\nORIGINAL QUERY:")
        report.append("-" * 40)
        report.append(results.get("original_query", "N/A"))
        
        report.append("\nOPTIMIZED QUERY:")
        report.append("-" * 40)
        report.append(results.get("optimized_query", "No optimization available"))
        
        if results.get("complexity_score"):
            report.append(f"\nCOMPLEXITY SCORE: {results['complexity_score']}/10")
        
        if results.get("final_recommendations"):
            report.append("\nRECOMMENDATIONS:")
            report.append("-" * 40)
            report.append(results["final_recommendations"])
        
        if results.get("index_recommendations"):
            report.append("\nINDEX RECOMMENDATIONS:")
            report.append("-" * 40)
            for i, rec in enumerate(results["index_recommendations"], 1):
                if isinstance(rec, dict):
                    if "raw_recommendation" in rec:
                        report.append(f"{i}. {rec['raw_recommendation']}")
                    else:
                        report.append(f"{i}. Table: {rec.get('table', 'Unknown')}")
                        report.append(f"   Columns: {', '.join(str(col) for col in rec.get('columns', []))}")
                else:
                    report.append(f"{i}. {str(rec)}")
        
        if results.get("optimization_steps"):
            report.append("\nOPTIMIZATION STEPS:")
            report.append("-" * 40)
            for i, step in enumerate(results["optimization_steps"], 1):
                report.append(f"{i}. {step}")
        
        if results.get("error"):
            report.append("\nERRORS:")
            report.append("-" * 40)
            report.append(results["error"])
        
        report.append("\n" + "="*80)
        return "\n".join(report)
    
    def _generate_html_report(self, results):
        """Generate HTML report"""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>SQL Query Optimization Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 10px; border-left: 5px solid #007cba; }}
                .section {{ margin: 20px 0; }}
                .code {{ background-color: #f8f8f8; padding: 10px; border: 1px solid #ddd; font-family: monospace; white-space: pre-wrap; }}
                .error {{ color: red; }}
                .metric {{ background-color: #e8f4f8; padding: 5px; margin: 5px 0; }}
            </style>
        </head>
        <body>
            <h1 class="header">SQL Query Optimization Report</h1>
            
            <div class="section">
                <h2>Original Query</h2>
                <div class="code">{results.get('original_query', 'N/A')}</div>
            </div>
            
            <div class="section">
                <h2>Optimized Query</h2>
                <div class="code">{results.get('optimized_query', 'No optimization available')}</div>
            </div>
            
            {f'<div class="section"><h2>Complexity Score</h2><div class="metric">Score: {results["complexity_score"]}/10</div></div>' if results.get('complexity_score') else ''}
            
            {f'<div class="section"><h2>Final Recommendations</h2><div>{results["final_recommendations"]}</div></div>' if results.get('final_recommendations') else ''}
            
            <div class="section">
                <h2>Index Recommendations</h2>
                <ul>
                    {''.join([f"<li>{rec}</li>" for rec in results.get('index_recommendations', [])])}
                </ul>
            </div>
            
            <div class="section">
                <h2>Optimization Steps</h2>
                <ol>
                    {''.join([f"<li>{step}</li>" for step in results.get('optimization_steps', [])])}
                </ol>
            </div>
            
            {f'<div class="section"><h2>Errors</h2><div class="error">{results["error"]}</div></div>' if results.get('error') else ''}
            
        </body>
        </html>
        """
        return html

def get_query_input():
    """
    Get SQL query from various sources in priority order:
    1. Sample query (if USE_SAMPLE_QUERY is true)
    2. stdin (if available)
    3. Interactive input
    
    Returns:
        str: SQL query to optimize
    """
    query = None
    
    # First check if we should use the sample query
    if CONFIG["USE_SAMPLE_QUERY"]:
        query = SAMPLE_QUERY.strip()
        logger.info("Using built-in sample query for demonstration")
        print("Using sample query:")
        print("-" * 40)
        print(query[:200] + "..." if len(query) > 200 else query)
        print("-" * 40)
        return query
    
    # If not using sample query, check stdin
    if not sys.stdin.isatty():
        query = sys.stdin.read().strip()
        if query:
            logger.info("Loaded query from stdin")
            return query
    
    # If still no query, prompt user interactively
    print("Enter SQL query (press Ctrl+D on a new line to finish):")
    print("Or set SQL_OPTIMIZER_USE_SAMPLE_QUERY=true to use the built-in sample query")
    query_lines = []
    try:
        while True:
            line = input()
            query_lines.append(line)
    except EOFError:
        query = "\n".join(query_lines).strip()
    
    return query

async def run_optimizer_cli():
    """Run the query optimizer using configuration from environment variables"""
    # Get query from various sources
    query = get_query_input()
    
    # If still no query, exit
    if not query:
        logger.error("No query provided. Exiting.")
        print("\nTo use the sample query, set: SQL_OPTIMIZER_USE_SAMPLE_QUERY=true")
        return
    
    # Prepare database connection parameters
    db_connection_params = None
    if all([CONFIG["DB_HOST"], CONFIG["DB_USER"], CONFIG["DB_PASSWORD"], CONFIG["DB_NAME"]]):
        db_connection_params = {
            "host": CONFIG["DB_HOST"],
            "user": CONFIG["DB_USER"],
            "password": CONFIG["DB_PASSWORD"],
            "database": CONFIG["DB_NAME"]
        }
        logger.info(f"Using database connection: {CONFIG['DB_HOST']}/{CONFIG['DB_NAME']}")
    else:
        logger.info("No database connection configured - using simulated plans")
    
    # Convert dialect string to enum
    dialect_map = {
        "mysql": DatabaseDialect.MYSQL,
        "postgresql": DatabaseDialect.POSTGRESQL,
        "sqlite": DatabaseDialect.SQLITE
    }
    dialect = dialect_map.get(CONFIG["DB_DIALECT"], DatabaseDialect.MYSQL)
    
    # Create and initialize optimizer using proper async context management
    async with QueryOptimizerWithLangGraph(
        server_path=CONFIG["SERVER_PATH"],
        host=CONFIG["SERVER_HOST"],
        port=CONFIG["SERVER_PORT"]
    ) as optimizer:
        
        # Initialize the optimizer
        await optimizer.initialize(db_connection_params)
        
        # Optimize the query
        logger.info(f"Optimizing query with dialect: {CONFIG['DB_DIALECT']}")
        result = await optimizer.optimize_query(query, dialect)
        
        # Generate report
        report = optimizer.generate_report(result, CONFIG["OUTPUT_FORMAT"])
        
        # Output report
        if CONFIG["OUTPUT_FILE"]:
            with open(CONFIG["OUTPUT_FILE"], "w") as f:
                f.write(report)
            logger.info(f"Report saved to {CONFIG['OUTPUT_FILE']}")
            # Print summary to console
            print(f"\nOptimization complete! Report saved to {CONFIG['OUTPUT_FILE']}")
            if CONFIG["OUTPUT_FORMAT"] == "html":
                print(f"Open the HTML report in a browser to view the full analysis.")
        else:
            print("\n" + report)

if __name__ == "__main__":
    asyncio.run(run_optimizer_cli())