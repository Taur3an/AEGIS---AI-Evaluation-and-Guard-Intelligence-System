"""
Dataset Loader Module for AEGIS

This module provides functionality for loading external/local prompt datasets
for adversarial testing, supporting Hugging Face datasets, CSV, and JSON files.
"""

import os
import json
import csv
import logging
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass, field

# Try to import optional dependencies
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

try:
    from datasets import load_dataset
    HF_DATASETS_AVAILABLE = True
except ImportError:
    HF_DATASETS_AVAILABLE = False
    load_dataset = None

logger = logging.getLogger(__name__)


@dataclass
class DatasetInfo:
    """Information about a loaded dataset."""
    name: str
    source: str  # file path or HF dataset name
    format: str  # csv, json, hf
    size: int  # number of records
    columns: List[str]  # available columns
    metadata: Dict[str, Any] = field(default_factory=dict)


class DatasetLoader:
    """Load external/local prompt datasets for adversarial testing."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the dataset loader.
        
        Args:
            config: Configuration dictionary with dataset loading options
        """
        self.config = config or {}
        self.loaded_datasets: Dict[str, DatasetInfo] = {}
        self._validate_dependencies()
    
    def _validate_dependencies(self) -> None:
        """Validate that required dependencies are available."""
        missing_deps = []
        if not PANDAS_AVAILABLE:
            missing_deps.append("pandas")
        if not HF_DATASETS_AVAILABLE:
            missing_deps.append("datasets (Hugging Face)")
        
        if missing_deps:
            logger.warning(f"Missing dependencies for DatasetLoader: {', '.join(missing_deps)}. "
                          "Install with: pip install pandas datasets")
    
    def load_dataset(
        self, 
        source: str, 
        dataset_name: Optional[str] = None,
        format_hint: Optional[str] = None
    ) -> DatasetInfo:
        """Load a dataset from various sources.
        
        Args:
            source: File path or Hugging Face dataset name
            dataset_name: Optional name to identify the dataset
            format_hint: Optional hint about the format ('csv', 'json', 'hf')
            
        Returns:
            DatasetInfo object with dataset information
        """
        # Determine the source type and format
        if self._is_huggingface_dataset(source):
            return self._load_huggingface_dataset(source, dataset_name)
        else:
            return self._load_local_dataset(source, dataset_name, format_hint)
    
    def _is_huggingface_dataset(self, source: str) -> bool:
        """Check if the source is a Hugging Face dataset."""
        # Simple heuristic: if it contains '/', it's likely a HF dataset
        return '/' in source and not os.path.exists(source)
    
    def _load_huggingface_dataset(
        self, 
        dataset_name: str, 
        user_defined_name: Optional[str] = None
    ) -> DatasetInfo:
        """Load a dataset from Hugging Face.
        
        Args:
            dataset_name: Hugging Face dataset name (e.g., 'imdb')
            user_defined_name: Optional name to identify the dataset
            
        Returns:
            DatasetInfo object with dataset information
        """
        if not HF_DATASETS_AVAILABLE:
            raise ImportError("Hugging Face datasets not available. Install with: pip install datasets")
        
        try:
            logger.info(f"Loading Hugging Face dataset: {dataset_name}")
            
            # Load the dataset
            dataset = load_dataset(dataset_name)
            
            # Get the first split (usually 'train')
            split_name = next(iter(dataset.keys()))
            data_split = dataset[split_name]
            
            # Convert to list of dictionaries
            records = [dict(row) for row in data_split]
            
            # Create dataset info
            info = DatasetInfo(
                name=user_defined_name or dataset_name,
                source=dataset_name,
                format="hf",
                size=len(records),
                columns=list(data_split.features.keys()) if data_split.features else [],
                metadata={
                    "splits": list(dataset.keys()),
                    "split_used": split_name,
                    "features": data_split.features if data_split.features else {}
                }
            )
            
            # Store the loaded data
            self.loaded_datasets[info.name] = info
            
            logger.info(f"Loaded HF dataset '{dataset_name}' with {len(records)} records")
            return info
            
        except Exception as e:
            logger.error(f"Failed to load Hugging Face dataset '{dataset_name}': {e}")
            raise
    
    def _load_local_dataset(
        self, 
        file_path: str, 
        dataset_name: Optional[str] = None,
        format_hint: Optional[str] = None
    ) -> DatasetInfo:
        """Load a local dataset file.
        
        Args:
            file_path: Path to the dataset file
            dataset_name: Optional name to identify the dataset
            format_hint: Optional hint about the format
            
        Returns:
            DatasetInfo object with dataset information
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
        # Determine format
        file_format = format_hint or self._detect_file_format(file_path)
        
        if file_format == "csv":
            return self._load_csv_dataset(file_path, dataset_name)
        elif file_format == "json":
            return self._load_json_dataset(file_path, dataset_name)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")
    
    def _detect_file_format(self, file_path: str) -> str:
        """Detect the file format based on extension."""
        _, ext = os.path.splitext(file_path.lower())
        if ext == ".csv":
            return "csv"
        elif ext in [".json", ".jsonl"]:
            return "json"
        else:
            raise ValueError(f"Unsupported file extension: {ext}")
    
    def _load_csv_dataset(self, file_path: str, dataset_name: Optional[str] = None) -> DatasetInfo:
        """Load a CSV dataset."""
        if not PANDAS_AVAILABLE:
            # Fallback to standard library CSV reader
            return self._load_csv_with_standard_lib(file_path, dataset_name)
        
        try:
            logger.info(f"Loading CSV dataset: {file_path}")
            
            # Load with pandas
            df = pd.read_csv(file_path)
            
            # Convert to list of dictionaries
            records = df.to_dict('records')
            
            # Create dataset info
            info = DatasetInfo(
                name=dataset_name or os.path.basename(file_path),
                source=file_path,
                format="csv",
                size=len(records),
                columns=list(df.columns),
                metadata={
                    "shape": df.shape,
                    "dtypes": df.dtypes.to_dict() if hasattr(df, 'dtypes') else {}
                }
            )
            
            # Store the loaded data
            self.loaded_datasets[info.name] = info
            
            logger.info(f"Loaded CSV dataset '{info.name}' with {len(records)} records")
            return info
            
        except Exception as e:
            logger.error(f"Failed to load CSV dataset '{file_path}': {e}")
            raise
    
    def _load_csv_with_standard_lib(self, file_path: str, dataset_name: Optional[str] = None) -> DatasetInfo:
        """Load CSV using standard library (fallback)."""
        try:
            logger.info(f"Loading CSV dataset with standard library: {file_path}")
            
            records = []
            columns = []
            
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                columns = reader.fieldnames or []
                
                for row in reader:
                    records.append(dict(row))
            
            # Create dataset info
            info = DatasetInfo(
                name=dataset_name or os.path.basename(file_path),
                source=file_path,
                format="csv",
                size=len(records),
                columns=columns,
                metadata={
                    "fallback_loader": True
                }
            )
            
            # Store the loaded data
            self.loaded_datasets[info.name] = info
            
            logger.info(f"Loaded CSV dataset '{info.name}' with {len(records)} records (standard lib)")
            return info
            
        except Exception as e:
            logger.error(f"Failed to load CSV dataset '{file_path}' with standard library: {e}")
            raise
    
    def _load_json_dataset(self, file_path: str, dataset_name: Optional[str] = None) -> DatasetInfo:
        """Load a JSON dataset."""
        try:
            logger.info(f"Loading JSON dataset: {file_path}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle different JSON structures
            if isinstance(data, list):
                # List of records
                records = data
                if records:
                    columns = list(records[0].keys()) if isinstance(records[0], dict) else []
                else:
                    columns = []
            elif isinstance(data, dict):
                # Dictionary with records
                if 'records' in data:
                    records = data['records']
                    columns = list(records[0].keys()) if records and isinstance(records[0], dict) else []
                elif 'data' in data:
                    records = data['data']
                    columns = list(records[0].keys()) if records and isinstance(records[0], dict) else []
                else:
                    # Assume the whole dict is one record
                    records = [data]
                    columns = list(data.keys())
            else:
                raise ValueError(f"Unsupported JSON structure in {file_path}")
            
            # Create dataset info
            info = DatasetInfo(
                name=dataset_name or os.path.basename(file_path),
                source=file_path,
                format="json",
                size=len(records),
                columns=columns,
                metadata={
                    "structure_type": type(data).__name__
                }
            )
            
            # Store the loaded data
            self.loaded_datasets[info.name] = info
            
            logger.info(f"Loaded JSON dataset '{info.name}' with {len(records)} records")
            return info
            
        except Exception as e:
            logger.error(f"Failed to load JSON dataset '{file_path}': {e}")
            raise
    
    def get_dataset(self, dataset_name: str) -> DatasetInfo:
        """Get information about a loaded dataset.
        
        Args:
            dataset_name: Name of the dataset to retrieve
            
        Returns:
            DatasetInfo object with dataset information
        """
        if dataset_name not in self.loaded_datasets:
            raise ValueError(f"Dataset '{dataset_name}' not found. Available datasets: {list(self.loaded_datasets.keys())}")
        
        return self.loaded_datasets[dataset_name]
    
    def list_datasets(self) -> List[str]:
        """List all loaded datasets.
        
        Returns:
            List of dataset names
        """
        return list(self.loaded_datasets.keys())
    
    def get_dataset_records(self, dataset_name: str) -> List[Dict[str, Any]]:
        """Get the actual records from a loaded dataset.
        
        Args:
            dataset_name: Name of the dataset to retrieve records from
            
        Returns:
            List of dataset records as dictionaries
        """
        # This would require storing the actual data, which we're not doing for memory efficiency
        # In a real implementation, this would return the actual dataset records
        info = self.get_dataset(dataset_name)
        logger.warning(f"get_dataset_records not implemented for dataset '{dataset_name}'. "
                      "This is a placeholder implementation.")
        return []
    
    def unload_dataset(self, dataset_name: str) -> bool:
        """Unload a dataset from memory.
        
        Args:
            dataset_name: Name of the dataset to unload
            
        Returns:
            True if successfully unloaded, False if not found
        """
        if dataset_name in self.loaded_datasets:
            del self.loaded_datasets[dataset_name]
            logger.info(f"Unloaded dataset '{dataset_name}'")
            return True
        return False
    
    def create_dataset_from_prompts(
        self, 
        prompts: List[str], 
        dataset_name: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> DatasetInfo:
        """Create a dataset from a list of prompts.
        
        Args:
            prompts: List of prompt strings
            dataset_name: Name for the new dataset
            metadata: Optional metadata for the dataset
            
        Returns:
            DatasetInfo object for the new dataset
        """
        # Create records from prompts
        records = [{"prompt": prompt, "id": i} for i, prompt in enumerate(prompts)]
        
        # Create dataset info
        info = DatasetInfo(
            name=dataset_name,
            source="manual_prompts",
            format="manual",
            size=len(records),
            columns=["prompt", "id"],
            metadata=metadata or {}
        )
        
        # Store the dataset info
        self.loaded_datasets[dataset_name] = info
        
        logger.info(f"Created manual dataset '{dataset_name}' with {len(records)} prompts")
        return info


# Convenience functions for common use cases
def load_prompts_dataset(
    source: str,
    dataset_name: Optional[str] = None,
    format_hint: Optional[str] = None
) -> DatasetInfo:
    """Convenience function to load a prompts dataset.
    
    Args:
        source: File path or Hugging Face dataset name
        dataset_name: Optional name to identify the dataset
        format_hint: Optional hint about the format
        
    Returns:
        DatasetInfo object with dataset information
    """
    loader = DatasetLoader()
    return loader.load_dataset(source, dataset_name, format_hint)


def create_prompts_dataset(
    prompts: List[str], 
    dataset_name: str,
    metadata: Optional[Dict[str, Any]] = None
) -> DatasetInfo:
    """Convenience function to create a prompts dataset.
    
    Args:
        prompts: List of prompt strings
        dataset_name: Name for the new dataset
        metadata: Optional metadata for the dataset
        
    Returns:
        DatasetInfo object for the new dataset
    """
    loader = DatasetLoader()
    return loader.create_dataset_from_prompts(prompts, dataset_name, metadata)


# CLI interface
def main():
    """CLI interface for the dataset loader."""
    import argparse
    
    parser = argparse.ArgumentParser(description="AEGIS Dataset Loader")
    parser.add_argument("source", help="File path or Hugging Face dataset name")
    parser.add_argument("--name", help="Dataset name")
    parser.add_argument("--format", help="File format hint (csv, json)")
    parser.add_argument("--list", action="store_true", help="List loaded datasets")
    
    args = parser.parse_args()
    
    loader = DatasetLoader()
    
    if args.list:
        datasets = loader.list_datasets()
        print("Loaded datasets:")
        for dataset in datasets:
            print(f"  - {dataset}")
        return
    
    try:
        info = loader.load_dataset(args.source, args.name, args.format)
        print(f"Dataset loaded successfully:")
        print(f"  Name: {info.name}")
        print(f"  Source: {info.source}")
        print(f"  Format: {info.format}")
        print(f"  Size: {info.size} records")
        print(f"  Columns: {', '.join(info.columns)}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())