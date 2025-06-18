#!/usr/bin/env python3
"""
Test script to verify the synthetic data generation setup for all three models.
"""

import os
import asyncio
from pathlib import Path
from dotenv import load_dotenv

# Test imports
try:
    import openai
    print("✅ OpenAI library imported successfully")
except ImportError:
    print("❌ OpenAI library not found. Run: pip install openai")
    exit(1)

try:
    import google.generativeai as genai
    print("✅ Google Generative AI library imported successfully")
except ImportError:
    print("❌ Google Generative AI not found. Run: pip install google-generativeai")
    exit(1)

try:
    import aiohttp
    print("✅ aiohttp library imported successfully")
except ImportError:
    print("❌ aiohttp not found. Run: pip install aiohttp")
    exit(1)

try:
    from config.settings import DEMOGRAPHICS, MODELS, PROCEDURES
    print("✅ Configuration loaded successfully")
except ImportError:
    print("❌ Configuration not found. Make sure config/settings.py exists")
    exit(1)

def test_api_keys():
    """Test if all required API keys are properly configured."""
    print("\n🔑 Testing API Keys:")
    
    # Load environment variables
    load_dotenv('.env')
    
    api_keys = {
        'OpenAI': os.getenv('OPENAI_API_KEY'),
        'Google': os.getenv('GOOGLE_API_KEY'),
        'xAI': os.getenv('XAI_API_KEY')
    }
    
    all_keys_present = True
    
    for service, key in api_keys.items():
        if key and key != 'your_api_key_here':
            print(f"✅ {service} API key found (ends with: ...{key[-4:]})")
        else:
            print(f"❌ {service} API key missing or not configured")
            all_keys_present = False
    
    if all_keys_present:
        print("✅ All API keys configured!")
    else:
        print("⚠️  Some API keys are missing. For full comparative analysis, all 3 are recommended.")
    
    return all_keys_present

def test_configuration():
    """Test if configuration is valid."""
    print("\n⚙️  Testing Configuration:")
    
    # Test demographics
    total_race = sum(DEMOGRAPHICS['race'].values())
    if abs(total_race - 1.0) < 0.01:
        print("✅ Race demographics sum to 1.0")
    else:
        print(f"⚠️  Race demographics sum to {total_race} (should be 1.0)")
    
    # Test models
    enabled_models = [model for model, config in MODELS.items() if config['enabled']]
    print(f"✅ Enabled models: {enabled_models}")
    
    if len(enabled_models) == 3:
        print("✅ All 3 models enabled for comparative analysis")
    else:
        print(f"⚠️  Only {len(enabled_models)}/3 models enabled")
    
    # Test procedures
    print(f"✅ Procedures configured: {len(PROCEDURES)} procedures")
    
    return True

async def test_generation():
    """Test note generation with available models."""
    print("\n🧪 Testing Note Generation:")
    
    try:
        from generate_notes import OperativeNoteGenerator
        
        generator = OperativeNoteGenerator()
        available_models = generator.check_api_keys()
        
        print(f"📋 Available models: {available_models}")
        
        if not available_models:
            print("❌ No models available for testing")
            return False
        
        # Test patient generation
        patient = generator.generate_patient_demographics()
        print(f"✅ Generated patient: {patient['age']}-year-old {patient['race']} {patient['gender']}")
        print(f"   Insurance: {patient['insurance']}, Language: {patient['primary_language']}")
        
        # Test prompt creation
        prompt = generator.create_prompt(patient)
        print(f"✅ Generated prompt ({len(prompt)} characters)")
        
        # Test note generation with first available model
        if available_models:
            test_model = available_models[0]
            print(f"🔄 Testing note generation with {test_model}...")
            
            note = await generator.generate_note(test_model, patient)
            
            if note:
                print("✅ Successfully generated a test note")
                print(f"   Note length: {len(note['operative_note'])} characters")
                print(f"   Complete: {note['validation']['is_complete']}")
                print(f"   Generation time: {note['metadata'].get('generation_time', 0):.2f}s")
                
                if note['validation']['missing_sections']:
                    print(f"   ⚠️  Missing sections: {note['validation']['missing_sections']}")
                
                if note['validation']['potential_bias_flags']:
                    print(f"   ⚠️  Potential bias flags: {note['validation']['potential_bias_flags']}")
                
                return True
            else:
                print("❌ Failed to generate test note")
                return False
        else:
            print("⚠️  No models available for testing")
            return False
            
    except Exception as e:
        print(f"❌ Error in generation test: {str(e)}")
        return False

def test_rate_limiting():
    """Test rate limiting configuration."""
    print("\n⏱️  Testing Rate Limiting:")
    
    from config.settings import RATE_LIMITS
    
    for model, limits in RATE_LIMITS.items():
        rpm = limits['requests_per_minute']
        delay = limits['delay_between_requests']
        print(f"✅ {model}: {rpm} req/min, {delay}s delay")
    
    print("✅ Rate limiting configured")
    return True

def main():
    """Run all tests."""
    print("🚀 Testing Multi-Model Synthetic Data Generation Setup")
    print("=" * 60)
    
    tests_passed = 0
    total_tests = 4
    
    # Test 1: API Keys
    if test_api_keys():
        tests_passed += 1
    
    # Test 2: Configuration
    if test_configuration():
        tests_passed += 1
    
    # Test 3: Rate Limiting
    if test_rate_limiting():
        tests_passed += 1
    
    # Test 4: Generation
    try:
        if asyncio.run(test_generation()):
            tests_passed += 1
    except Exception as e:
        print(f"❌ Generation test failed: {str(e)}")
    
    # Summary
    print(f"\n📊 Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! Ready to generate synthetic notes with all models.")
        print("\nTo start generation, run:")
        print("   python generate_notes.py")
        print("\nThis will generate 1,000 notes per available model for comparative bias analysis.")
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
        
        if not Path('.env').exists():
            print("\n💡 Setup checklist:")
            print("   1. Rename 'api_keys.env' to '.env'")
            print("   2. Add your API keys to the .env file:")
            print("      - OpenAI API key (for GPT-4)")
            print("      - Google API key (for Gemini)")
            print("      - xAI API key (for Grok)")
            print("   3. Run: pip install -r requirements.txt")

if __name__ == "__main__":
    main() 