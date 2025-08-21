"""
LM Studio Enhancement for AEGIS
Adds support for uncensored local models via LM Studio

This enhancement extends the existing AEGIS framework to support LM Studio,
which provides an excellent platform for running uncensored models locally
for red teaming and adversarial testing.
"""

import asyncio
import json
import logging
import requests
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

# Enhanced imports for LM Studio support
try:
    import requests
    LMSTUDIO_AVAILABLE = True
except ImportError:
    LMSTUDIO_AVAILABLE = False
    print("Requests not installed. Install with: pip install requests")


@dataclass
class LMStudioConfig:
    """Configuration specifically for LM Studio connections."""
    host: str = "localhost"
    port: int = 1234
    model_name: str = "local-model"
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    timeout: int = 30
    
    @property
    def base_url(self) -> str:
        """Get the full LM Studio API URL."""
        return f"http://{self.host}:{self.port}/v1"


class LMStudioProvider:
    """
    LM Studio provider for running uncensored models locally.
    
    LM Studio provides an OpenAI-compatible API for local model inference,
    making it ideal for red teaming with uncensored models that don't have
    built-in safety filters.
    
    Recommended models for red teaming:
    - WizardLM-Uncensored variants
    - Dolphin models (uncensored)
    - CodeLlama-Instruct variants
    - Mixtral-Instruct uncensored variants
    """
    
    def __init__(self, config: LMStudioConfig):
        self.config = config
        self.session = requests.Session()
        self.session.timeout = config.timeout
        
    async def generate_completion(self, prompt: str, **kwargs) -> str:
        """Generate text completion using LM Studio."""
        try:
            # Prepare the request payload (OpenAI-compatible format)
            payload = {
                "model": self.config.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
                "temperature": kwargs.get("temperature", self.config.temperature),
                "top_p": kwargs.get("top_p", self.config.top_p),
                "stream": False
            }
            
            # Make the API call
            response = await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: self.session.post(
                    f"{self.config.base_url}/chat/completions",
                    json=payload,
                    headers={"Content-Type": "application/json"}
                )
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                raise Exception(f"LM Studio API error: {response.status_code} - {response.text}")
                
        except Exception as e:
            logging.error(f"LM Studio generation failed: {e}")
            raise
    
    async def check_health(self) -> bool:
        """Check if LM Studio server is available."""
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.session.get(f"{self.config.base_url}/models", timeout=5)
            )
            return response.status_code == 200
        except:
            return False
    
    async def list_models(self) -> List[str]:
        """List available models in LM Studio."""
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.session.get(f"{self.config.base_url}/models")
            )
            if response.status_code == 200:
                models_data = response.json()
                return [model["id"] for model in models_data.get("data", [])]
            return []
        except:
            return []


# Enhanced LLMConfig to support LM Studio
def create_lmstudio_config(
    host: str = "localhost",
    port: int = 1234,
    model_name: str = "local-model",
    temperature: float = 0.7
):
    """Create an LLMConfig for LM Studio provider."""
    from dataclasses import dataclass
    
    @dataclass
    class LLMConfig:
        provider: str = "lmstudio"
        model_name: str = model_name
        api_key: Optional[str] = None
        base_url: str = f"http://{host}:{port}/v1"
        temperature: float = temperature
        max_tokens: int = 2048
        
    return LLMConfig()


# Integration code for existing BaseLLM class
def enhance_base_llm_with_lmstudio():
    """
    Code to add to the existing BaseLLM._initialize_client method:
    
    Add this elif clause after the existing providers:
    """
    code_snippet = '''
            elif self.config.provider == "lmstudio" and LMSTUDIO_AVAILABLE:
                # LM Studio uses OpenAI-compatible API
                self.client = openai.OpenAI(
                    api_key="not-needed",  # LM Studio doesn't require API key
                    base_url=self.config.base_url or "http://localhost:1234/v1"
                )
    '''
    return code_snippet


def enhance_generate_response_with_lmstudio():
    """
    Code to add to existing generate_response methods for LM Studio support:
    """
    code_snippet = '''
            elif self.config.provider == "lmstudio":
                try:
                    response = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: self.client.chat.completions.create(
                            model=self.config.model_name,
                            messages=[{"role": "user", "content": prompt}],
                            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
                            temperature=kwargs.get("temperature", self.config.temperature)
                        )
                    )
                    return response.choices[0].message.content
                except Exception as e:
                    logging.error(f"LM Studio generation failed: {e}")
                    return "Error: Failed to generate response with LM Studio"
    '''
    return code_snippet


# Uncensored model recommendations
RECOMMENDED_UNCENSORED_MODELS = {
    "red_teaming": [
        "WizardLM-13B-Uncensored",
        "WizardLM-30B-Uncensored", 
        "dolphin-2.2.1-mistral-7b",
        "dolphin-2.6-mixtral-8x7b",
        "CodeLlama-13B-Instruct",
        "Mixtral-8x7B-Instruct-v0.1-uncensored"
    ],
    "reasoning": [
        "OpenHermes-2.5-Mistral-7B",
        "Neural-Chat-7B-v3-1",
        "Starling-LM-7B-alpha"
    ],
    "creative": [
        "MythoMax-L2-13B",
        "Airoboros-L2-13B-2.1"
    ]
}


def get_lmstudio_setup_guide():
    """Return setup instructions for LM Studio with AEGIS."""
    return """
# LM Studio Setup Guide for AEGIS

## 1. Download and Install LM Studio
- Visit: https://lmstudio.ai/
- Download for your platform (Windows, macOS, Linux)
- Install and launch LM Studio

## 2. Download Recommended Models
For red teaming, download these uncensored models:

### High Priority (Start Here):
- **WizardLM-13B-Uncensored** - Excellent for adversarial prompts
- **dolphin-2.2.1-mistral-7b** - Fast and effective for attacks
- **Mixtral-8x7B-Instruct-uncensored** - Powerful reasoning for complex attacks

### Advanced Models:
- **WizardLM-30B-Uncensored** - More capable but requires more VRAM
- **CodeLlama-13B-Instruct** - Great for code-based attacks

## 3. Configure LM Studio
1. Load your chosen model in LM Studio
2. Go to "Local Server" tab
3. Start the server (default: localhost:1234)
4. Ensure "Cross-Origin-Resource-Sharing (CORS)" is enabled

## 4. Test Connection
```python
# Test LM Studio connection
config = create_lmstudio_config(
    host="localhost",
    port=1234,
    model_name="your-loaded-model-name",
    temperature=0.8  # Higher temperature for more creative attacks
)

# Create attacker with LM Studio
attacker = AttackerLLM(config)
```

## 5. Optimal Settings for Red Teaming
- **Temperature**: 0.7-0.9 (higher for more creative attacks)
- **Top-p**: 0.9 (allows diverse token selection)
- **Max Tokens**: 2048-4096 (longer responses for complex attacks)

## 6. Security Considerations
- Run LM Studio on isolated networks for sensitive testing
- Monitor resource usage - uncensored models can be computationally intensive
- Keep attack logs secure and follow responsible disclosure practices
"""


# Example usage patterns
def get_usage_examples():
    """Return example usage patterns for LM Studio with AEGIS."""
    return """
# Example 1: Basic LM Studio Configuration
config = create_lmstudio_config(
    host="localhost",
    port=1234,
    model_name="WizardLM-13B-Uncensored",
    temperature=0.8
)

# Example 2: High-intensity Red Teaming Setup
attacker_config = create_lmstudio_config(
    host="localhost", 
    port=1234,
    model_name="dolphin-2.6-mixtral-8x7b",
    temperature=0.9  # High creativity for novel attacks
)

defender_config = create_lmstudio_config(
    host="localhost",
    port=1235,  # Different port if running multiple models
    model_name="Mixtral-8x7B-Instruct-v0.1",
    temperature=0.3  # Lower temperature for consistent defense
)

# Example 3: Multi-Model Setup
attacker = AttackerLLM(attacker_config)
defender = DefenderLLM(defender_config) 
judge = JudgeLLM(create_lmstudio_config(model_name="OpenHermes-2.5-Mistral-7B"))

# Example 4: Evaluation with LM Studio
orchestrator = RedTeamOrchestrator(attacker, defender, judge)
evaluation_target = RewardHackingTarget()

results = await orchestrator.evaluate_target(
    evaluation_target,
    num_iterations=10,
    attack_strategies=["goal_substitution", "reward_gaming", "specification_gaming"]
)
"""


if __name__ == "__main__":
    print("LM Studio Enhancement for AEGIS")
    print("=" * 50)
    print(get_lmstudio_setup_guide())
    print("\nUsage Examples:")
    print("=" * 50)
    print(get_usage_examples())