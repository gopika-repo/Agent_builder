"""
Table Reasoning Agent

Advanced table understanding with:
- Pandas-based analysis
- Aggregations (max, min, sum, avg)
- Trend detection
- Cross-table comparisons
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import re

logger = logging.getLogger(__name__)

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


@dataclass
class TableAnalysisResult:
    """Result of table analysis"""
    table_id: str
    operation: str
    result: Any
    explanation: str
    confidence: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "table_id": self.table_id,
            "operation": self.operation,
            "result": self.result,
            "explanation": self.explanation,
            "confidence": self.confidence
        }


class TableReasoningAgent:
    """
    Advanced table reasoning capabilities.
    
    Supports:
    - Max/min value finding
    - Aggregations (sum, average)
    - Trend detection
    - Comparisons between cells
    - Cross-table analysis
    """
    
    def __init__(self):
        self.operations = {
            "max": self._find_max,
            "min": self._find_min,
            "sum": self._calculate_sum,
            "avg": self._calculate_average,
            "average": self._calculate_average,
            "trend": self._detect_trend,
            "compare": self._compare_values,
            "count": self._count_rows,
            "find": self._find_value
        }
    
    def analyze(
        self,
        table_data: Dict[str, Any],
        query: str,
        table_id: str = ""
    ) -> TableAnalysisResult:
        """
        Analyze a table based on a natural language query.
        
        Args:
            table_data: Table with headers and rows
            query: Natural language query about the table
            table_id: Identifier for the table
            
        Returns:
            TableAnalysisResult with analysis output
        """
        # Parse query to determine operation
        operation, column, params = self._parse_query(query, table_data)
        
        if not PANDAS_AVAILABLE:
            return self._fallback_analysis(table_data, operation, column, table_id)
        
        # Convert to DataFrame
        df = self._to_dataframe(table_data)
        
        if df.empty:
            return TableAnalysisResult(
                table_id=table_id,
                operation=operation,
                result=None,
                explanation="Table is empty or could not be parsed",
                confidence=0.0
            )
        
        # Execute operation
        if operation in self.operations:
            return self.operations[operation](df, column, params, table_id)
        else:
            return self._general_analysis(df, query, table_id)
    
    def _parse_query(
        self,
        query: str,
        table_data: Dict[str, Any]
    ) -> Tuple[str, Optional[str], Dict]:
        """Parse query to extract operation and target column"""
        query_lower = query.lower()
        
        # Detect operation
        operation = "general"
        
        if any(w in query_lower for w in ["maximum", "highest", "largest", "max"]):
            operation = "max"
        elif any(w in query_lower for w in ["minimum", "lowest", "smallest", "min"]):
            operation = "min"
        elif any(w in query_lower for w in ["total", "sum"]):
            operation = "sum"
        elif any(w in query_lower for w in ["average", "mean", "avg"]):
            operation = "avg"
        elif any(w in query_lower for w in ["trend", "increasing", "decreasing", "growth"]):
            operation = "trend"
        elif "compare" in query_lower:
            operation = "compare"
        elif any(w in query_lower for w in ["count", "how many"]):
            operation = "count"
        elif "find" in query_lower or "where" in query_lower:
            operation = "find"
        
        # Find target column
        headers = table_data.get("headers", [])
        column = None
        
        for header in headers:
            if header.lower() in query_lower:
                column = header
                break
        
        # If no column found, try to guess from context
        if not column:
            # Look for numeric-sounding columns for aggregations
            for header in headers:
                if any(w in header.lower() for w in ["revenue", "amount", "total", "value", "price", "count", "profit"]):
                    column = header
                    break
        
        return operation, column, {}
    
    def _to_dataframe(self, table_data: Dict[str, Any]) -> 'pd.DataFrame':
        """Convert table data to pandas DataFrame"""
        headers = table_data.get("headers", [])
        rows = table_data.get("rows", [])
        
        if not headers or not rows:
            return pd.DataFrame()
        
        try:
            df = pd.DataFrame(rows, columns=headers)
            
            # Try to convert numeric columns
            for col in df.columns:
                df[col] = pd.to_numeric(
                    df[col].astype(str).str.replace(r'[$,€£%]', '', regex=True),
                    errors='ignore'
                )
            
            return df
        except Exception as e:
            logger.warning(f"DataFrame conversion failed: {e}")
            return pd.DataFrame()
    
    def _find_max(
        self,
        df: 'pd.DataFrame',
        column: Optional[str],
        params: Dict,
        table_id: str
    ) -> TableAnalysisResult:
        """Find maximum value"""
        if column and column in df.columns:
            try:
                max_val = df[column].max()
                max_row = df[df[column] == max_val].iloc[0].to_dict()
                
                return TableAnalysisResult(
                    table_id=table_id,
                    operation="max",
                    result={"value": max_val, "row": max_row},
                    explanation=f"The maximum value in '{column}' is {max_val}",
                    confidence=0.95
                )
            except Exception as e:
                logger.warning(f"Max operation failed: {e}")
        
        # Try all numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            col = numeric_cols[0]
            max_val = df[col].max()
            return TableAnalysisResult(
                table_id=table_id,
                operation="max",
                result={"value": max_val, "column": col},
                explanation=f"The maximum value in '{col}' is {max_val}",
                confidence=0.8
            )
        
        return TableAnalysisResult(
            table_id=table_id,
            operation="max",
            result=None,
            explanation="Could not find numeric column for max operation",
            confidence=0.0
        )
    
    def _find_min(
        self,
        df: 'pd.DataFrame',
        column: Optional[str],
        params: Dict,
        table_id: str
    ) -> TableAnalysisResult:
        """Find minimum value"""
        if column and column in df.columns:
            try:
                min_val = df[column].min()
                min_row = df[df[column] == min_val].iloc[0].to_dict()
                
                return TableAnalysisResult(
                    table_id=table_id,
                    operation="min",
                    result={"value": min_val, "row": min_row},
                    explanation=f"The minimum value in '{column}' is {min_val}",
                    confidence=0.95
                )
            except Exception:
                pass
        
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            col = numeric_cols[0]
            min_val = df[col].min()
            return TableAnalysisResult(
                table_id=table_id,
                operation="min",
                result={"value": min_val, "column": col},
                explanation=f"The minimum value in '{col}' is {min_val}",
                confidence=0.8
            )
        
        return TableAnalysisResult(
            table_id=table_id,
            operation="min",
            result=None,
            explanation="Could not find numeric column for min operation",
            confidence=0.0
        )
    
    def _calculate_sum(
        self,
        df: 'pd.DataFrame',
        column: Optional[str],
        params: Dict,
        table_id: str
    ) -> TableAnalysisResult:
        """Calculate sum of column"""
        if column and column in df.columns:
            try:
                total = df[column].sum()
                return TableAnalysisResult(
                    table_id=table_id,
                    operation="sum",
                    result={"total": total, "column": column},
                    explanation=f"The sum of '{column}' is {total}",
                    confidence=0.95
                )
            except Exception:
                pass
        
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            col = numeric_cols[0]
            total = df[col].sum()
            return TableAnalysisResult(
                table_id=table_id,
                operation="sum",
                result={"total": total, "column": col},
                explanation=f"The sum of '{col}' is {total}",
                confidence=0.8
            )
        
        return TableAnalysisResult(
            table_id=table_id,
            operation="sum",
            result=None,
            explanation="Could not find numeric column for sum",
            confidence=0.0
        )
    
    def _calculate_average(
        self,
        df: 'pd.DataFrame',
        column: Optional[str],
        params: Dict,
        table_id: str
    ) -> TableAnalysisResult:
        """Calculate average of column"""
        if column and column in df.columns:
            try:
                avg = df[column].mean()
                return TableAnalysisResult(
                    table_id=table_id,
                    operation="average",
                    result={"average": round(avg, 2), "column": column},
                    explanation=f"The average of '{column}' is {avg:.2f}",
                    confidence=0.95
                )
            except Exception:
                pass
        
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            col = numeric_cols[0]
            avg = df[col].mean()
            return TableAnalysisResult(
                table_id=table_id,
                operation="average",
                result={"average": round(avg, 2), "column": col},
                explanation=f"The average of '{col}' is {avg:.2f}",
                confidence=0.8
            )
        
        return TableAnalysisResult(
            table_id=table_id,
            operation="average",
            result=None,
            explanation="Could not find numeric column for average",
            confidence=0.0
        )
    
    def _detect_trend(
        self,
        df: 'pd.DataFrame',
        column: Optional[str],
        params: Dict,
        table_id: str
    ) -> TableAnalysisResult:
        """Detect trend in data (increasing/decreasing)"""
        if column and column in df.columns:
            try:
                values = df[column].dropna().values
                if len(values) < 2:
                    return TableAnalysisResult(
                        table_id=table_id,
                        operation="trend",
                        result=None,
                        explanation="Not enough data points for trend analysis",
                        confidence=0.0
                    )
                
                # Calculate trend
                first_half = values[:len(values)//2].mean()
                second_half = values[len(values)//2:].mean()
                
                if second_half > first_half * 1.05:
                    trend = "increasing"
                    change = ((second_half - first_half) / first_half) * 100
                elif second_half < first_half * 0.95:
                    trend = "decreasing"
                    change = ((first_half - second_half) / first_half) * 100
                else:
                    trend = "stable"
                    change = 0
                
                return TableAnalysisResult(
                    table_id=table_id,
                    operation="trend",
                    result={
                        "trend": trend,
                        "change_percent": round(abs(change), 2),
                        "column": column
                    },
                    explanation=f"The '{column}' shows a {trend} trend with {abs(change):.1f}% change",
                    confidence=0.85
                )
            except Exception as e:
                logger.warning(f"Trend detection failed: {e}")
        
        return TableAnalysisResult(
            table_id=table_id,
            operation="trend",
            result=None,
            explanation="Could not detect trend",
            confidence=0.0
        )
    
    def _compare_values(
        self,
        df: 'pd.DataFrame',
        column: Optional[str],
        params: Dict,
        table_id: str
    ) -> TableAnalysisResult:
        """Compare values in a column"""
        if column and column in df.columns:
            try:
                stats = {
                    "min": df[column].min(),
                    "max": df[column].max(),
                    "mean": df[column].mean(),
                    "range": df[column].max() - df[column].min()
                }
                
                return TableAnalysisResult(
                    table_id=table_id,
                    operation="compare",
                    result=stats,
                    explanation=f"'{column}' ranges from {stats['min']} to {stats['max']} (mean: {stats['mean']:.2f})",
                    confidence=0.9
                )
            except Exception:
                pass
        
        return TableAnalysisResult(
            table_id=table_id,
            operation="compare",
            result=None,
            explanation="Could not compare values",
            confidence=0.0
        )
    
    def _count_rows(
        self,
        df: 'pd.DataFrame',
        column: Optional[str],
        params: Dict,
        table_id: str
    ) -> TableAnalysisResult:
        """Count rows in table"""
        count = len(df)
        return TableAnalysisResult(
            table_id=table_id,
            operation="count",
            result={"count": count},
            explanation=f"The table has {count} rows",
            confidence=1.0
        )
    
    def _find_value(
        self,
        df: 'pd.DataFrame',
        column: Optional[str],
        params: Dict,
        table_id: str
    ) -> TableAnalysisResult:
        """Find specific value or row"""
        if column and column in df.columns:
            sample = df[column].iloc[0] if len(df) > 0 else None
            return TableAnalysisResult(
                table_id=table_id,
                operation="find",
                result={"column": column, "sample": sample, "count": len(df)},
                explanation=f"Found {len(df)} entries in '{column}'",
                confidence=0.8
            )
        
        return TableAnalysisResult(
            table_id=table_id,
            operation="find",
            result=None,
            explanation="Could not find specified value",
            confidence=0.0
        )
    
    def _general_analysis(
        self,
        df: 'pd.DataFrame',
        query: str,
        table_id: str
    ) -> TableAnalysisResult:
        """General table analysis"""
        stats = {
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": list(df.columns),
            "numeric_columns": list(df.select_dtypes(include=['number']).columns)
        }
        
        return TableAnalysisResult(
            table_id=table_id,
            operation="general",
            result=stats,
            explanation=f"Table has {stats['rows']} rows and {stats['columns']} columns",
            confidence=0.7
        )
    
    def _fallback_analysis(
        self,
        table_data: Dict[str, Any],
        operation: str,
        column: Optional[str],
        table_id: str
    ) -> TableAnalysisResult:
        """Fallback analysis without pandas"""
        headers = table_data.get("headers", [])
        rows = table_data.get("rows", [])
        
        return TableAnalysisResult(
            table_id=table_id,
            operation=operation,
            result={
                "headers": headers,
                "row_count": len(rows),
                "note": "Pandas not available for advanced analysis"
            },
            explanation=f"Table has {len(headers)} columns and {len(rows)} rows",
            confidence=0.5
        )
    
    def compare_tables(
        self,
        table1: Dict[str, Any],
        table2: Dict[str, Any],
        column: str
    ) -> TableAnalysisResult:
        """Compare values between two tables"""
        if not PANDAS_AVAILABLE:
            return TableAnalysisResult(
                table_id="comparison",
                operation="cross_compare",
                result=None,
                explanation="Pandas required for cross-table comparison",
                confidence=0.0
            )
        
        df1 = self._to_dataframe(table1)
        df2 = self._to_dataframe(table2)
        
        if column not in df1.columns or column not in df2.columns:
            return TableAnalysisResult(
                table_id="comparison",
                operation="cross_compare",
                result=None,
                explanation=f"Column '{column}' not found in both tables",
                confidence=0.0
            )
        
        comparison = {
            "table1_sum": df1[column].sum(),
            "table2_sum": df2[column].sum(),
            "difference": df1[column].sum() - df2[column].sum(),
            "percent_diff": ((df1[column].sum() - df2[column].sum()) / df2[column].sum() * 100)
            if df2[column].sum() != 0 else 0
        }
        
        return TableAnalysisResult(
            table_id="comparison",
            operation="cross_compare",
            result=comparison,
            explanation=f"Table 1 {column}: {comparison['table1_sum']}, "
                       f"Table 2: {comparison['table2_sum']} "
                       f"({comparison['percent_diff']:.1f}% difference)",
            confidence=0.9
        )
