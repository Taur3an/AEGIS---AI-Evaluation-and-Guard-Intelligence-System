#!/usr/bin/env python3
"""
AEGIS Command Line Interface

This CLI provides a unified interface for all AEGIS functionalities,
including risk evaluation, adversarial prompt generation, and comprehensive testing.

Usage:
    python -m aegis.cli --help
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from typing import List, Dict, Any, Optional
from pathlib import Path

# Import AEGIS components
from . import (
    initialize_aegis,
    get_system_status,
    get_supported_risk_categories,
    list_available_attacks,
    evaluate_single_risk,
    evaluate_comprehensive_risk,
    RiskCategory
)
from .modules.dataset_loader import DatasetLoader, load_prompts_dataset, create_prompts_dataset
from .modules.uncensored_model_loader import (
    UncensoredModelLoader, 
    ModelConfig, 
    ModelProvider,
    create_uncensored_config
)
from .modules.adversarial_prompt_generator import (
    AdversarialPromptGenerator, 
    PromptGenerationConfig,
    create_prompt_generator
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_parser() -> argparse.ArgumentParser:
    """Set up the main argument parser."""
    parser = argparse.ArgumentParser(
        prog='aegis',
        description='AEGIS - AI Evaluation and Guard Intelligence System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  aegis init                              # Initialize AEGIS system
  aegis status                            # Check system status
  aegis evaluate --risk reward_hacking    # Evaluate single risk category
  aegis comprehensive                     # Run comprehensive risk evaluation
  aegis generate --category deception     # Generate adversarial prompts
  aegis list --attacks                    # List available attack vectors
  aegis load --dataset ./prompts.csv      # Load prompt dataset
        """
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Init command
    init_parser = subparsers.add_parser('init', help='Initialize AEGIS system')
    init_parser.add_argument(
        '--config',
        help='Path to configuration file'
    )
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Check system status')
    
    # Evaluate command
    evaluate_parser = subparsers.add_parser('evaluate', help='Evaluate single risk category')
    evaluate_parser.add_argument(
        '--risk', '-r',
        required=True,
        choices=[cat.value for cat in RiskCategory],
        help='Risk category to evaluate'
    )
    evaluate_parser.add_argument(
        '--prompt', '-p',
        required=True,
        help='Input prompt to evaluate'
    )
    evaluate_parser.add_argument(
        '--response', '-r',
        required=True,
        help='AI response to evaluate'
    )
    evaluate_parser.add_argument(
        '--output', '-o',
        help='Output file for results (JSON format)'
    )
    
    # Comprehensive evaluation command
    comprehensive_parser = subparsers.add_parser('comprehensive', help='Run comprehensive risk evaluation')
    comprehensive_parser.add_argument(
        '--prompt', '-p',
        required=True,
        help='Input prompt to evaluate'
    )
    comprehensive_parser.add_argument(
        '--response', '-r',
        required=True,
        help='AI response to evaluate'
    )
    comprehensive_parser.add_argument(
        '--output', '-o',
        help='Output file for results (JSON format)'
    )
    
    # Generate command
    generate_parser = subparsers.add_parser('generate', help='Generate adversarial prompts')
    generate_parser.add_argument(
        '--category', '-c',
        choices=[cat.value for cat in RiskCategory],
        help='Risk category to generate prompts for (all if not specified)'
    )
    generate_parser.add_argument(
        '--count', '-n',
        type=int,
        default=10,
        help='Number of prompts to generate per category'
    )
    generate_parser.add_argument(
        '--context',
        help='Context for prompt generation'
    )
    generate_parser.add_argument(
        '--output', '-o',
        help='Output directory for generated prompts'
    )
    generate_parser.add_argument(
        '--model-provider',
        choices=['ollama', 'lm_studio', 'local'],
        default='ollama',
        help='Model provider for generation'
    )
    generate_parser.add_argument(
        '--model-name',
        default='llama3:8b',
        help='Model name for generation'
    )
    
    # List command
    list_parser = subparsers.add_parser('list', help='List system information')
    list_parser.add_argument(
        '--attacks', '-a',
        action='store_true',
        help='List available attack vectors'
    )
    list_parser.add_argument(
        '--categories', '-c',
        action='store_true',
        help='List supported risk categories'
    )
    list_parser.add_argument(
        '--datasets', '-d',
        action='store_true',
        help='List loaded datasets'
    )
    
    # Load command
    load_parser = subparsers.add_parser('load', help='Load datasets or models')
    load_parser.add_argument(
        '--dataset', '-d',
        help='Path to dataset file (CSV, JSON, or HF dataset name)'
    )
    load_parser.add_argument(
        '--model', '-m',
        help='Load uncensored model'
    )
    load_parser.add_argument(
        '--provider',
        choices=['ollama', 'lm_studio', 'local'],
        default='ollama',
        help='Model provider'
    )
    
    return parser


def cmd_init(args: argparse.Namespace) -> int:
    """Initialize AEGIS system."""
    print("Initializing AEGIS system...")
    
    try:
        status = initialize_aegis(args.config)
        if status['status'] == 'initialized':
            print("✅ AEGIS initialized successfully!")
            print(f"   Version: {status['version']}")
            print(f"   Risk Categories: {status['risk_categories_count']}")
            print(f"   Components: {', '.join([k for k, v in status['components'].items() if v])}")
            return 0
        else:
            print(f"❌ Failed to initialize AEGIS: {status.get('error', 'Unknown error')}")
            return 1
    except Exception as e:
        print(f"❌ Error during initialization: {e}")
        return 1


def cmd_status(args: argparse.Namespace) -> int:
    """Check system status."""
    try:
        status = get_system_status()
        print("AEGIS System Status:")
        print(f"  Initialized: {'✅ Yes' if status['initialized'] else '❌ No'}")
        print(f"  Version: {status.get('version', 'Unknown')}")
        print(f"  Risk Categories: {status.get('risk_categories_count', 0)}")
        
        if status['components']:
            print("  Components:")
            for component, available in status['components'].items():
                status_icon = '✅' if available else '❌'
                print(f"    {component}: {status_icon}")
        
        return 0
    except Exception as e:
        print(f"❌ Error checking status: {e}")
        return 1


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Evaluate single risk category."""
    try:
        # Convert risk category string to enum
        risk_category = RiskCategory(args.risk.lower())
        
        # Evaluate risk
        print(f"Evaluating {args.risk} risk...")
        assessment = evaluate_single_risk(
            risk_category,
            args.prompt,
            args.response
        )
        
        # Display results
        print(f"\n📊 Risk Assessment Results:")
        print(f"  Risk Score: {assessment.overall_risk_score:.3f}")
        print(f"  Risk Level: {assessment.risk_level.value}")
        print(f"  Vulnerabilities: {len(assessment.vulnerability_flags)}")
        
        if assessment.vulnerability_flags:
            print("  Vulnerability Flags:")
            for flag in assessment.vulnerability_flags[:5]:  # Show first 5
                print(f"    - {flag}")
            if len(assessment.vulnerability_flags) > 5:
                print(f"    ... and {len(assessment.vulnerability_flags) - 5} more")
        
        # Export results if requested
        if args.output:
            result_data = {
                "risk_category": args.risk,
                "prompt": args.prompt,
                "response": args.response,
                "overall_risk_score": assessment.overall_risk_score,
                "risk_level": assessment.risk_level.value,
                "vulnerability_flags": assessment.vulnerability_flags,
                "detailed_analysis": assessment.detailed_analysis,
                "metadata": assessment.metadata
            }
            
            with open(args.output, 'w') as f:
                json.dump(result_data, f, indent=2)
            
            print(f"\n💾 Results saved to: {args.output}")
        
        return 0
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        return 1


def cmd_comprehensive(args: argparse.Namespace) -> int:
    """Run comprehensive risk evaluation."""
    try:
        print("Running comprehensive risk evaluation...")
        results = evaluate_comprehensive_risk(
            args.prompt,
            args.response
        )
        
        # Display results
        overall = results['overall_analysis']
        print(f"\n📊 Comprehensive Risk Assessment:")
        print(f"  Average Risk Score: {overall['average_risk_score']:.3f}")
        print(f"  Maximum Risk Score: {overall['maximum_risk_score']:.3f}")
        print(f"  Risk Level: {overall['risk_level']}")
        print(f"  Categories Evaluated: {overall['total_categories_evaluated']}")
        
        if overall['high_risk_categories']:
            print(f"  High-Risk Categories: {', '.join(overall['high_risk_categories'])}")
        
        # Show category breakdown
        print(f"\n📈 Category Breakdown:")
        for category, details in results['category_breakdown'].items():
            print(f"  {category}: {details['risk_score']:.3f} ({details['risk_level']})")
        
        # Export results if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"\n💾 Results saved to: {args.output}")
        
        return 0
    except Exception as e:
        print(f"❌ Error during comprehensive evaluation: {e}")
        return 1


async def cmd_generate(args: argparse.Namespace) -> int:
    """Generate adversarial prompts."""
    try:
        print(f"Generating adversarial prompts using {args.model_provider}...")
        
        # Create prompt generator
        generator = create_prompt_generator(
            model_provider=args.model_provider,
            model_name=args.model_name,
            prompt_count=args.count
        )
        
        # Set output directory
        if args.output:
            generator.config.export_directory = args.output
        else:
            generator.config.export_directory = "./generated_prompts/"
        
        # Ensure directory exists
        Path(generator.config.export_directory).mkdir(parents=True, exist_ok=True)
        
        # Generate prompts
        if args.category:
            # Generate for specific category
            risk_category = RiskCategory(args.category.lower())
            print(f"Generating prompts for {risk_category.value}...")
            prompts = await generator.generate_prompts_for_risk_category(
                risk_category,
                args.context,
                args.count
            )
        else:
            # Generate comprehensive suite
            print("Generating comprehensive prompt suite...")
            prompts = await generator.generate_comprehensive_prompt_suite(
                context=args.context,
                prompts_per_category=args.count
            )
        
        # Export prompts
        exported_files = generator.export_prompts(prompts)
        
        # Display results
        print(f"\n✅ Generated {len(prompts)} adversarial prompts")
        print(f"   Export Directory: {generator.config.export_directory}")
        
        if exported_files:
            print("   Exported Files:")
            for format_name, file_path in exported_files.items():
                print(f"     {format_name.upper()}: {file_path}")
        
        return 0
    except Exception as e:
        print(f"❌ Error during prompt generation: {e}")
        return 1


def cmd_list(args: argparse.Namespace) -> int:
    """List system information."""
    try:
        if args.attacks:
            attacks = list_available_attacks()
            print(f"🛡️ Available Attack Vectors ({len(attacks)}):")
            for attack in attacks[:20]:  # Show first 20
                print(f"  - {attack.get('name', 'Unknown')} ({attack.get('category', 'Unknown')})")
            if len(attacks) > 20:
                print(f"  ... and {len(attacks) - 20} more")
        
        if args.categories:
            categories = get_supported_risk_categories()
            print(f"📁 Supported Risk Categories ({len(categories)}):")
            for category in categories:
                print(f"  - {category.value}")
        
        if args.datasets:
            # This would require a dataset loader instance
            print("📂 Loaded Datasets: None (dataset loading not yet fully implemented)")
        
        if not any([args.attacks, args.categories, args.datasets]):
            # Show all if no specific option selected
            cmd_list(argparse.Namespace(attacks=True, categories=True, datasets=True))
        
        return 0
    except Exception as e:
        print(f"❌ Error listing information: {e}")
        return 1


def cmd_load(args: argparse.Namespace) -> int:
    """Load datasets or models."""
    try:
        if args.dataset:
            print(f"Loading dataset: {args.dataset}")
            # This would use the DatasetLoader
            loader = DatasetLoader()
            dataset_info = loader.load_dataset(args.dataset)
            print(f"✅ Dataset loaded successfully:")
            print(f"   Name: {dataset_info.name}")
            print(f"   Format: {dataset_info.format}")
            print(f"   Records: {dataset_info.size}")
            print(f"   Columns: {', '.join(dataset_info.columns)}")
        
        if args.model:
            print(f"Loading model: {args.model} via {args.provider}")
            # This would use the UncensoredModelLoader
            model_config = create_uncensored_config(
                provider=args.provider,
                model_name=args.model
            )
            loader = UncensoredModelLoader()
            client = loader.load_model(model_config)
            print(f"✅ Model loaded successfully: {args.model}")
        
        if not any([args.dataset, args.model]):
            print("Please specify what to load (--dataset or --model)")
            return 1
        
        return 0
    except Exception as e:
        print(f"❌ Error during loading: {e}")
        return 1


async def main_async() -> int:
    """Async main function."""
    parser = setup_parser()
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Handle commands
    if args.command == 'init':
        return cmd_init(args)
    elif args.command == 'status':
        return cmd_status(args)
    elif args.command == 'evaluate':
        return cmd_evaluate(args)
    elif args.command == 'comprehensive':
        return cmd_comprehensive(args)
    elif args.command == 'generate':
        return await cmd_generate(args)
    elif args.command == 'list':
        return cmd_list(args)
    elif args.command == 'load':
        return cmd_load(args)
    else:
        # Show help if no command specified
        parser.print_help()
        return 0


def main() -> int:
    """Main entry point."""
    try:
        return asyncio.run(main_async())
    except KeyboardInterrupt:
        print("\n👋 AEGIS CLI interrupted by user")
        return 130  # Standard exit code for Ctrl+C
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())