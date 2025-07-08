from typing import Any
import httpx
from mcp.server.fastmcp import FastMCP

import os
import re
import sys
import json
import enum
import logging
import argparse
from typing import Optional, List, Dict, Any, Union

# Official MCP imports
import mcp
from mcp.server.models import InitializationOptions
from mcp.types import (
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
    LoggingLevel
)

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("query_optimizer_server")

# Import optional database modules
try:
    import mysql.connector
    DATABASE_SUPPORT = True
except ImportError:
    logger.warning("MySQL support not available")
    logger.warning("Install with: pip install mysql-connector-python")
    DATABASE_SUPPORT = False

# Try to load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    logger.warning("python-dotenv not installed, not loading .env file")
    logger.warning("Install with: pip install python-dotenv")




# Initialize FastMCP server
mcp = FastMCP("queryoptimizer")



class DatabaseDialect(str, enum.Enum):
    """Supported database dialects"""
    MYSQL = "mysql"
    POSTGRESQL = "postgresql"
    SQLITE = "sqlite"

class DatabaseConnection:
    """
    Database connection wrapper for different dialects
    """
    
    def __init__(self, dialect: DatabaseDialect, connection_params: dict):
        """
        Initialize a database connection
        
        Args:
            dialect: Database dialect
            connection_params: Connection parameters
        """
        self.dialect = dialect
        self.connection_params = connection_params
        self.connection = None
    
    def connect(self):
        """Establish a database connection"""
        if self.dialect == DatabaseDialect.MYSQL:
            if not DATABASE_SUPPORT:
                return {"error": "MySQL support not available"}
            
            try:
                self.connection = mysql.connector.connect(**self.connection_params)
                return True
            except Exception as e:
                return {"error": f"MySQL connection failed: {str(e)}"}
        elif self.dialect == DatabaseDialect.POSTGRESQL:
            return {"error": "PostgreSQL support not implemented yet"}
        elif self.dialect == DatabaseDialect.SQLITE:
            return {"error": "SQLite support not implemented yet"}
        else:
            return {"error": f"Unsupported dialect: {self.dialect}"}
    
    def execute_query(self, query: str) -> dict:
        """
        Execute a query
        
        Args:
            query: SQL query to execute
            
        Returns:
            Query results or error message
        """
        if not self.connection:
            connect_result = self.connect()
            if isinstance(connect_result, dict) and "error" in connect_result:
                return connect_result
        
        try:
            cursor = self.connection.cursor(dictionary=True)
            cursor.execute(query)
            results = cursor.fetchall()
            cursor.close()
            return {"results": results}
        except Exception as e:
            return {"error": f"Query execution failed: {str(e)}"}
    
    def get_explain_plan(self, query: str) -> dict:
        """
        Get EXPLAIN plan for a query
        
        Args:
            query: SQL query to explain
            
        Returns:
            Explain plan or error message
        """
        if not self.connection:
            connect_result = self.connect()
            if isinstance(connect_result, dict) and "error" in connect_result:
                return connect_result
        
        try:
            explain_query = f"EXPLAIN {query}"
            cursor = self.connection.cursor(dictionary=True)
            cursor.execute(explain_query)
            results = cursor.fetchall()
            cursor.close()
            return {"explain_plan": results}
        except Exception as e:
            return {"error": f"Explain plan failed: {str(e)}"}
    
    def get_existing_indexes(self, table: str) -> dict:
        """
        Get existing indexes for a table
        
        Args:
            table: Table name
            
        Returns:
            Index information or error message
        """
        if not self.connection:
            connect_result = self.connect()
            if isinstance(connect_result, dict) and "error" in connect_result:
                return connect_result
        
        try:
            if self.dialect == DatabaseDialect.MYSQL:
                query = f"SHOW INDEX FROM {table}"
                cursor = self.connection.cursor(dictionary=True)
                cursor.execute(query)
                results = cursor.fetchall()
                cursor.close()
                return {"indexes": results}
            else:
                return {"error": f"Index retrieval not implemented for {self.dialect}"}
        except Exception as e:
            return {"error": f"Index retrieval failed: {str(e)}"}
    
    def close(self):
        """Close the database connection"""
        if self.connection:
            self.connection.close()

def get_db_connection(dialect: DatabaseDialect, connection_params: dict) -> DatabaseConnection:
    """
    Create a database connection
    
    Args:
        dialect: Database dialect
        connection_params: Connection parameters
        
    Returns:
        Database connection object
    """
    return DatabaseConnection(dialect, connection_params)

def get_llm():
    """
    Get LLM instance with configured settings
    
    Returns:
        LLM instance
    """
    # Get API key from environment
    api_key = os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        logger.warning("OPENAI_API_KEY not found in environment, using default")
    
    # Create LLM with reasonable settings
    llm = ChatOpenAI(
        model="gpt-4o",  # Use a capable model
        temperature=0.1,  # Almost deterministic for technical analysis
        api_key=api_key,
        timeout=60       # Give it time for complex queries
    )
    
    return llm

class DatabaseDialect(str, enum.Enum):
    """Supported database dialects"""
    MYSQL = "mysql"
    POSTGRESQL = "postgresql"
    SQLITE = "sqlite"

class DatabaseConnection:
    """
    Database connection wrapper for different dialects
    """
    
    def __init__(self, dialect: DatabaseDialect, connection_params: dict):
        """
        Initialize a database connection
        
        Args:
            dialect: Database dialect
            connection_params: Connection parameters
        """
        self.dialect = dialect
        self.connection_params = connection_params
        self.connection = None
    
    def connect(self):
        """Establish a database connection"""
        if self.dialect == DatabaseDialect.MYSQL:
            if not DATABASE_SUPPORT:
                return {"error": "MySQL support not available"}
            
            try:
                self.connection = mysql.connector.connect(**self.connection_params)
                return True
            except Exception as e:
                return {"error": f"MySQL connection failed: {str(e)}"}
        elif self.dialect == DatabaseDialect.POSTGRESQL:
            return {"error": "PostgreSQL support not implemented yet"}
        elif self.dialect == DatabaseDialect.SQLITE:
            return {"error": "SQLite support not implemented yet"}
        else:
            return {"error": f"Unsupported dialect: {self.dialect}"}
    
    def execute_query(self, query: str) -> dict:
        """
        Execute a query
        
        Args:
            query: SQL query to execute
            
        Returns:
            Query results or error message
        """
        if not self.connection:
            connect_result = self.connect()
            if isinstance(connect_result, dict) and "error" in connect_result:
                return connect_result
        
        try:
            cursor = self.connection.cursor(dictionary=True)
            cursor.execute(query)
            results = cursor.fetchall()
            cursor.close()
            return {"results": results}
        except Exception as e:
            return {"error": f"Query execution failed: {str(e)}"}
    
    def get_explain_plan(self, query: str) -> dict:
        """
        Get EXPLAIN plan for a query
        
        Args:
            query: SQL query to explain
            
        Returns:
            Explain plan or error message
        """
        if not self.connection:
            connect_result = self.connect()
            if isinstance(connect_result, dict) and "error" in connect_result:
                return connect_result
        
        try:
            explain_query = f"EXPLAIN {query}"
            cursor = self.connection.cursor(dictionary=True)
            cursor.execute(explain_query)
            results = cursor.fetchall()
            cursor.close()
            return {"explain_plan": results}
        except Exception as e:
            return {"error": f"Explain plan failed: {str(e)}"}
    
    def get_existing_indexes(self, table: str) -> dict:
        """
        Get existing indexes for a table
        
        Args:
            table: Table name
            
        Returns:
            Index information or error message
        """
        if not self.connection:
            connect_result = self.connect()
            if isinstance(connect_result, dict) and "error" in connect_result:
                return connect_result
        
        try:
            if self.dialect == DatabaseDialect.MYSQL:
                query = f"SHOW INDEX FROM {table}"
                cursor = self.connection.cursor(dictionary=True)
                cursor.execute(query)
                results = cursor.fetchall()
                cursor.close()
                return {"indexes": results}
            else:
                return {"error": f"Index retrieval not implemented for {self.dialect}"}
        except Exception as e:
            return {"error": f"Index retrieval failed: {str(e)}"}
    
    def close(self):
        """Close the database connection"""
        if self.connection:
            self.connection.close()

def get_db_connection(dialect: DatabaseDialect, connection_params: dict) -> DatabaseConnection:
    """
    Create a database connection
    
    Args:
        dialect: Database dialect
        connection_params: Connection parameters
        
    Returns:
        Database connection object
    """
    return DatabaseConnection(dialect, connection_params)

def get_llm():
    """
    Get LLM instance with configured settings
    
    Returns:
        LLM instance
    """
    # Get API key from environment
    api_key = os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        logger.warning("OPENAI_API_KEY not found in environment, using default")
    
    # Create LLM with reasonable settings
    llm = ChatOpenAI(
        model="gpt-4o",  # Use a capable model
        temperature=0.1,  # Almost deterministic for technical analysis
        api_key=api_key,
        timeout=60       # Give it time for complex queries
    )
    
    return llm

#############################################
##              MCP SERVER TOOLS           ##
#############################################

@mcp.tool()
def extract_tables(state: dict) -> dict:
    """
    Extract tables used in the SQL query
    
    Args:
        state: Current optimization state
        
    Returns:
        Updated state with extracted tables
    """
    query = state["query"]
    
    try:
        # Get LLM
        model = get_llm()
        
        # Create extract tables prompt
        extract_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a SQL expert. Identify all tables used in SQL queries.
            
            Example query:
            SELECT c.customer_name, o.order_date
            FROM customers c
            INNER JOIN orders o ON c.customer_id = o.customer_id
            WHERE o.order_date > '2023-01-01'
            
            For the above example, the output would be:
            ["customers", "orders"]
            
            Return ONLY a JSON array of table names, without aliases.
            """),
            ("human", "{query}")
        ])
        
        # Execute the LLM chain
        chain = extract_prompt | model | StrOutputParser()
        response = chain.invoke({"query": query})
        
        # Extract the tables from the response
        # Find a valid JSON array in the response
        tables_match = re.search(r'\[(.*?)\]', response, re.DOTALL)
        if tables_match:
            tables_json = f"[{tables_match.group(1)}]"
            try:
                tables = json.loads(tables_json)
                # Ensure all tables are strings
                tables = [str(table).strip('"\'').strip() for table in tables]
            except:
                # If JSON parsing fails, try basic regex
                tables = [t.strip('"\'').strip() for t in re.findall(r'["\'"]?([\w_]+)["\'"]?', tables_match.group(1))]
        else:
            # Fallback to regex if no JSON array is found
            tables = re.findall(r'(?:FROM|JOIN)\s+([a-zA-Z0-9_]+)', query, re.IGNORECASE)
            tables = list(set(tables))  # Remove duplicates
        
        updated_state = {
            **state,
            "tables": tables,
            "optimization_steps": state.get("optimization_steps", []) + ["Extracted tables"],
            "next_step": "analyze_complexity"
        }
        
        return updated_state
        
    except Exception as e:
        updated_state = {
            **state,
            "error": f"Table extraction failed: {str(e)}",
            "optimization_steps": state.get("optimization_steps", []) + ["Table extraction failed"],
            "next_step": "analyze_complexity"
        }
        
        return updated_state
    

@mcp.tool()
def analyze_complexity(state: dict) -> dict:
    """
    Analyze query complexity and score it
    
    Args:
        state: Current optimization state
        
    Returns:
        Updated state with complexity analysis
    """
    query = state["query"]
    
    try:
        # Get LLM
        model = get_llm()
        
        # Create complexity analysis prompt
        complexity_prompt = ChatPromptTemplate.from_messages([
            ("system", """Analyze SQL query complexity on a scale of 1-10, where:
            1: Simple query with one table and no joins
            10: Highly complex query with multiple joins, subqueries, window functions, etc.
            
            Return only a JSON object with:
            - score: integer 1-10
            - analysis: brief analysis of complexity factors
            - improvement_potential: "high", "medium", or "low"
            """),
            ("human", "{query}")
        ])
        
        # Execute the LLM chain
        chain = complexity_prompt | model | StrOutputParser()
        response = chain.invoke({"query": query})
        
        # Try to extract JSON from the response
        try:
            # Find JSON object in response
            json_match = re.search(r'({.*})', response, re.DOTALL)
            if json_match:
                complexity_data = json.loads(json_match.group(1))
            else:
                # Fallback if no JSON found
                complexity_data = {
                    "score": 5,  # Default medium complexity
                    "analysis": "Unable to parse complexity analysis",
                    "improvement_potential": "medium"
                }
        except:
            # If JSON parsing fails, try to extract score with regex
            score_match = re.search(r'score["\']?\s*:\s*(\d+)', response)
            score = int(score_match.group(1)) if score_match else 5
            
            # Extract analysis with regex
            analysis_match = re.search(r'analysis["\']?\s*:\s*["\'](.*?)["\']', response)
            analysis = analysis_match.group(1) if analysis_match else "Analysis not available"
            
            # Extract improvement potential with regex
            potential_match = re.search(r'improvement_potential["\']?\s*:\s*["\'](.*?)["\']', response)
            potential = potential_match.group(1) if potential_match else "medium"
            
            complexity_data = {
                "score": score,
                "analysis": analysis,
                "improvement_potential": potential
            }
        
        updated_state = {
            **state,
            "complexity_score": complexity_data.get("score", 5),
            "complexity_analysis": complexity_data.get("analysis", ""),
            "improvement_potential": complexity_data.get("improvement_potential", "medium"),
            "optimization_steps": state.get("optimization_steps", []) + ["Analyzed query complexity"],
            "next_step": "get_explain_plan"
        }
        
        return updated_state
        
    except Exception as e:
        updated_state = {
            **state,
            "error": f"Complexity analysis failed: {str(e)}",
            "optimization_steps": state.get("optimization_steps", []) + ["Complexity analysis failed"],
            "next_step": "get_explain_plan"
        }
        
        return updated_state

@mcp.tool()
def get_explain_plan(state: dict, db_connection_params: Optional[dict] = None) -> dict:
    """
    Generate an EXPLAIN plan for the query
    
    Args:
        state: Current optimization state
        db_connection_params: Optional database connection parameters
        
    Returns:
        Updated state with explain plan
    """
    query = state["query"]
    dialect = state["dialect"]
    
    # Try to get actual EXPLAIN plan if DB connection available
    if DATABASE_SUPPORT and db_connection_params:
        try:
            db_connection = get_db_connection(dialect, db_connection_params)
            result = db_connection.get_explain_plan(query)
            db_connection.close()
            
            if not isinstance(result, dict) or "error" not in result:
                updated_state = {
                    **state,
                    "explain_plan": json.dumps(result, indent=2),
                    "optimization_steps": state.get("optimization_steps", []) + ["Retrieved actual explain plan"],
                    "next_step": "analyze_explain_plan"
                }
                
                return updated_state
        except Exception as e:
            logger.error(f"Failed to get actual explain plan: {str(e)}")
            # Continue with simulated plan
    
    # If we're here, either no DB connection or it failed, generate simulated plan
    try:
        # Get LLM
        model = get_llm()
        
        # Create explain plan prompt
        explain_prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are a {dialect} database expert. Generate a detailed and realistic EXPLAIN plan 
            for the SQL query. Format as if it came directly from the database. Include:
            
            1. Execution steps
            2. Table access methods
            3. Join operations
            4. Estimated costs and rows
            5. Any potential bottlenecks
            
            Make it as realistic as possible for a {dialect} database.
            """),
            ("human", "{query}")
        ])
        
        # Execute the LLM chain
        chain = explain_prompt | model
        response = chain.invoke({"query": query})
        
        updated_state = {
            **state,
            "explain_plan": response.content,
            "optimization_steps": state.get("optimization_steps", []) + ["Generated simulated explain plan"],
            "next_step": "analyze_explain_plan"
        }
        
        return updated_state
        
    except Exception as e:
        updated_state = {
            **state,
            "error": f"Explain plan generation failed: {str(e)}",
            "optimization_steps": state.get("optimization_steps", []) + ["Explain plan generation failed"],
            "next_step": "analyze_explain_plan"
        }
        
        return updated_state

@mcp.tool()
def analyze_explain_plan(state: dict) -> dict:
    """
    Analyze the explain plan and identify optimization opportunities
    
    Args:
        state: Current optimization state with explain plan
        
    Returns:
        Updated state with explain plan analysis
    """
    query = state["query"]
    explain_plan = state.get("explain_plan", "")
    
    try:
        # Get LLM
        model = get_llm()
        
        # Create analysis prompt
        analyze_prompt = ChatPromptTemplate.from_messages([
            ("system", """Analyze the database EXPLAIN plan output:
            1. Identify performance bottlenecks
            2. Evaluate index usage
            3. Identify table scans or costly operations
            4. Suggest query optimizations based on the plan
            5. Provide a summary of key findings"""),
            ("human", "Analyze explain plan: {explain_plan}\n\nFor query: {query}")
        ])
        
        chain = analyze_prompt | model
        response = chain.invoke({
            "explain_plan": explain_plan,
            "query": query
        })
        
        updated_state = {
            **state,
            "current_analysis": response.content,
            "optimization_steps": state.get("optimization_steps", []) + ["Analyzed explain plan"],
            "next_step": "recommend_indexes"
        }
        
        return updated_state
        
    except Exception as e:
        updated_state = {
            **state,
            "error": f"Explain plan analysis failed: {str(e)}",
            "optimization_steps": state.get("optimization_steps", []) + ["Explain plan analysis failed"],
            "next_step": "recommend_indexes"
        }
        
        return updated_state

@mcp.tool()
def recommend_indexes(state: dict, db_connection_params: Optional[dict] = None) -> dict:
    """
    Generate index recommendations based on query analysis
    
    Args:
        state: Current optimization state
        db_connection_params: Optional database connection parameters
        
    Returns:
        Updated state with index recommendations
    """
    query = state["query"]
    tables = state.get("tables", [])
    dialect = state["dialect"]
    
    # Get existing indexes if we have DB connection
    existing_indexes_info = "No database connection available"
    if DATABASE_SUPPORT and db_connection_params and tables:
        try:
            db_connection = get_db_connection(dialect, db_connection_params)
            existing_indexes = {}
            for table in tables:
                indexes = db_connection.get_existing_indexes(table)
                if not isinstance(indexes, dict) or 'error' not in indexes:
                    existing_indexes[table] = indexes
            existing_indexes_info = str(existing_indexes)
        except Exception as e:
            existing_indexes_info = f"Error retrieving existing indexes: {str(e)}"
    
    try:
        # Get LLM
        model = get_llm()
        
        # Create index recommendation prompt
        index_prompt = ChatPromptTemplate.from_messages([
            ("system", """Recommend database indexes to optimize query performance:
            1. Analyze query predicates and join conditions
            2. Suggest composite or single-column indexes
            3. Provide rationale for each recommendation
            4. Consider trade-offs in index creation
            5. Consider existing indexes: {existing_indexes}"""),
            ("human", "Recommend indexes for query: {query}")
        ])
        
        chain = index_prompt | model
        response = chain.invoke({
            "query": query,
            "existing_indexes": existing_indexes_info
        })
        
        # Parse recommendations (simplified)
        index_pattern = r'(CREATE\s+INDEX|Recommended\s+Index).*?on\s+(\w+)\s*\(([^)]+)\)'
        indexes = []
        
        for match in re.finditer(index_pattern, response.content, re.IGNORECASE):
            indexes.append({
                "type": match.group(1),
                "table": match.group(2),
                "columns": [col.strip() for col in match.group(3).split(',')]
            })
        
        # If no structured recommendations found, return raw text
        if not indexes and response.content.strip():
            indexes.append({
                "raw_recommendation": response.content.strip()
            })
        
        updated_state = {
            **state,
            "index_recommendations": indexes,
            "optimization_steps": state.get("optimization_steps", []) + ["Generated index recommendations"],
            "next_step": "optimize_query"
        }
        
        return updated_state
        
    except Exception as e:
        updated_state = {
            **state,
            "error": f"Index recommendation failed: {str(e)}",
            "index_recommendations": [{"error": f"Index recommendation failed: {str(e)}"}],
            "optimization_steps": state.get("optimization_steps", []) + ["Index recommendation failed"],
            "next_step": "optimize_query"
        }
        
        return updated_state

@mcp.tool()
def optimize_query(state: dict) -> dict:
    """
    Apply dialect-specific optimizations to the query
    
    Args:
        state: Current optimization state
        
    Returns:
        Updated state with optimized query
    """
    query = state["query"]
    dialect = state["dialect"]
    
    try:
        # Get LLM
        model = get_llm()
        
        # Create optimization prompt
        optimization_prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are an expert {dialect} database query optimizer. Optimize the SQL query for better performance:
            
            Apply these optimizations as appropriate:
            1. Convert old-style comma joins to explicit INNER JOIN syntax
            2. Replace nested subqueries with JOINs or CTEs where possible
            3. Optimize WHERE clause conditions
            4. Improve GROUP BY and ORDER BY operations
            5. Apply {dialect}-specific optimizations
            
            Return ONLY the optimized SQL query, no explanations."""),
            ("human", "Optimize this SQL query: {query}")
        ])
        
        chain = optimization_prompt | model
        response = chain.invoke({"query": query})
        
        # Extract SQL from response
        response_text = response.content
        
        # Try to find SQL in code blocks
        sql_match = re.search(r"```(?:sql)?\s*\n(.*?)\n\s*```", response_text, re.DOTALL)
        if sql_match:
            optimized_query = sql_match.group(1).strip()
        else:
            # Use the entire response and clean it up
            optimized_query = response_text.strip()
            optimized_query = optimized_query.replace("```sql", "").replace("```", "").strip()
        
        # Clean up the query
        sql_keywords = ["SELECT", "WITH", "UPDATE", "DELETE", "INSERT"]
        for keyword in sql_keywords:
            keyword_match = re.search(rf"(?i){keyword}\s", optimized_query)
            if keyword_match:
                start_index = keyword_match.start()
                optimized_query = optimized_query[start_index:]
                break
        
        # Ensure query ends with semicolon
        if not optimized_query.endswith(';'):
            optimized_query += ';'
        
        updated_state = {
            **state,
            "final_optimized_query": optimized_query,
            "optimization_steps": state.get("optimization_steps", []) + ["Applied dialect-specific optimizations"],
            "next_step": "generate_final_recommendations"
        }
        
        return updated_state
        
    except Exception as e:
        updated_state = {
            **state,
            "error": f"Query optimization failed: {str(e)}",
            "optimization_steps": state.get("optimization_steps", []) + ["Query optimization failed"],
            "next_step": "generate_final_recommendations"
        }
        
        return updated_state

@mcp.tool()
def generate_final_recommendations(state: dict) -> dict:
    """
    Generate final optimization recommendations summary
    
    Args:
        state: Current optimization state
        
    Returns:
        Updated state with final recommendations
    """
    query = state["query"]
    complexity_score = state.get("complexity_score")
    current_analysis = state.get("current_analysis", "")
    explain_plan = state.get("explain_plan", "")
    index_recommendations = state.get("index_recommendations", [])
    optimized_query = state.get("final_optimized_query")
    
    try:
        # Get LLM
        model = get_llm()
        
        # Create final recommendations prompt
        final_prompt = ChatPromptTemplate.from_messages([
            ("system", """Generate comprehensive SQL optimization recommendations:
            1. Summarize key findings from analysis
            2. Explain performance improvements
            3. Detail index recommendations
            4. Address potential trade-offs
            5. Provide actionable next steps"""),
            ("human", """Original query: {query}
            
            Complexity score: {complexity_score}
            Analysis: {current_analysis}
            Explain plan: {explain_plan}
            Index recommendations: {index_recommendations}
            Optimized query: {optimized_query}
            
            Provide comprehensive recommendations.""")
        ])
        
        chain = final_prompt | model
        response = chain.invoke({
            "query": query,
            "complexity_score": complexity_score,
            "current_analysis": current_analysis,
            "explain_plan": explain_plan,
            "index_recommendations": str(index_recommendations),
            "optimized_query": optimized_query
        })
        
        updated_state = {
            **state,
            "final_recommendations": response.content,
            "optimization_steps": state.get("optimization_steps", []) + ["Generated final recommendations"],
            "next_step": "complete"
        }
        
        return updated_state
        
    except Exception as e:
        updated_state = {
            **state,
            "error": f"Final recommendations generation failed: {str(e)}",
            "optimization_steps": state.get("optimization_steps", []) + ["Final recommendations generation failed"],
            "next_step": "complete"
        }
        
        return updated_state
    
def main():
    """Run the server"""
    
    # Configure logging
    
    
    logger.info(f"Starting SQL Query Optimizer MCP server on ")
    
    try:
        # Run MCP server
       #mcp.run()
       mcp.run(transport="stdio")
    except KeyboardInterrupt:
        logger.info("Server stopped")

if __name__ == "__main__":
    main()