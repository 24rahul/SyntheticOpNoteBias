# Synthetic Data Generation

This module generates synthetic operative notes using **three AI models** (GPT-4, Gemini, Grok) for comprehensive comparative bias analysis in medical documentation across different patient demographics.

## 📋 Overview

The synthetic data generation system creates realistic operative notes with systematically varied patient demographics across **three foundation models** to enable comparative bias analysis:

**Models for Comparison:**
- **GPT-4 Turbo** (OpenAI)
- **Gemini Pro** (Google)  
- **Grok** (xAI)

**Demographics:**
- **Race**: White, Black, Asian, Native American, Other
- **Gender**: Male, Female, Non-binary  
- **Ethnicity**: Hispanic, Non-Hispanic
- **Insurance**: Medicare, Medicaid, Private Insurance
- **Language**: English, Spanish, Other languages
- **Medical factors**: Various procedures, pain levels, and comorbidities

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set Up API Keys
Rename `api_keys.env` to `.env` and add **ALL THREE** API keys:
```bash
mv api_keys.env .env
# Edit .env file with your actual API keys:
# - OPENAI_API_KEY (for GPT-4)
# - GOOGLE_API_KEY (for Gemini)  
# - XAI_API_KEY (for Grok)
```

**Get API Keys:**
- OpenAI: https://platform.openai.com/api-keys
- Google: https://aistudio.google.com/app/apikey
- xAI: https://x.ai/api

### 3. Test Setup
```bash
python test_setup.py
```

### 4. Generate Notes
```bash
python generate_notes.py
```

This will generate **1,000 notes per model** (3,000 total) for comparative analysis.

## 📁 Directory Structure

```
1_Synthetic_Data_Generation/
├── config/
│   └── settings.py          # Configuration parameters
├── output/                  # Generated notes and metadata
├── logs/                    # Generation logs
├── generate_notes.py        # Main generation script
├── test_setup.py           # Setup verification script
├── requirements.txt        # Python dependencies
├── api_keys.env           # API keys template
└── README.md              # This file
```

## ⚙️ Configuration

### Demographics (config/settings.py)
- **Equal distribution** across racial groups (20% each)
- **Gender distribution**: 40% male, 40% female, 20% non-binary
- **Insurance mix**: 33% Medicare, 33% Medicaid, 34% Private
- **Language distribution**: 60% English, 20% Spanish, 20% Other

### Medical Parameters
- **Age range**: 18-80 years
- **Procedures**: 5 common surgeries (appendectomy, cholecystectomy, CABG, knee replacement, hysterectomy)
- **Pain levels**: Mild, moderate, severe
- **Comorbidities**: Up to 3 random conditions per patient

### Generation Settings
- **1,000 notes per model** (3,000 total)
- **All 3 models required** for comparative analysis
- **Rate limiting**: Model-specific delays to respect API limits
- **Parallel generation**: Each model generates independently

## 📊 Output Format

### Generated Notes
Each note includes:
```json
{
  "patient_demographics": {
    "age": 45,
    "race": "Black",
    "gender": "female",
    "ethnicity": "non-Hispanic",
    "insurance": "Medicaid",
    "primary_language": "English",
    "procedure": "cholecystectomy",
    "pain_level": "moderate",
    "comorbidities": ["diabetes mellitus", "hypertension"]
  },
  "operative_note": "Complete surgical note text...",
  "validation": {
    "is_complete": true,
    "missing_sections": [],
    "has_demographics": true,
    "has_pain_management": true,
    "potential_bias_flags": []
  },
  "metadata": {
    "model": "gpt4",
    "timestamp": "2024-01-15T10:30:00",
    "tokens_used": 850,
    "generation_time": 2.34
  }
}
```

### Model-Specific Output Files
- **GPT-4**: `synthetic_notes_gpt4_{timestamp}.json`
- **Gemini**: `synthetic_notes_gemini_{timestamp}.json`
- **Grok**: `synthetic_notes_grok_{timestamp}.json`

### Comparative Metadata
Summary statistics for each model including:
- **Generation info**: Model config, success rates, performance metrics
- **Demographics distribution**: Actual vs. target distributions per model
- **Validation summary**: Completion rates, bias flags per model

## 🔍 Quality Assurance

### Automatic Validation
Each generated note is validated for:
- **Completeness**: All required sections present
- **Demographics inclusion**: Patient details documented
- **Pain management**: Appropriate pain assessment and treatment
- **Bias detection**: Flags potentially biased language

### Model-Specific Rate Limiting
- **GPT-4**: 3,500 requests/min (high throughput)
- **Gemini**: 60 requests/min (conservative)
- **Grok**: 100 requests/min (moderate)

### Bias Detection Keywords
The system flags notes containing potentially problematic terms:
- "non-compliant", "drug-seeking", "exaggerating pain"
- "malingering", "difficult patient", "uncooperative"
- "poor historian"

## 📈 Monitoring

### Real-time Logging
- **Progress tracking**: Notes generated per model
- **Error logging**: Failed generations with reasons
- **Performance metrics**: Generation time, token usage per model
- **Rate limiting**: Automatic throttling for API compliance

### Output Files
- **Notes**: `synthetic_notes_{model}_{timestamp}.json`
- **Metadata**: `metadata_{model}_{timestamp}.json`  
- **Logs**: `logs/generation_{timestamp}.log`

## 🔧 Customization

### Adjusting Model Settings
Edit `config/settings.py` to modify model parameters:
```python
MODELS = {
    'gpt4': {
        'temperature': 0.7,    # Adjust creativity
        'max_tokens': 4000,    # Response length
        'top_p': 0.9          # Nucleus sampling
    }
    # ... other models
}
```

### Rate Limiting
Modify generation speed in `config/settings.py`:
```python
RATE_LIMITS = {
    'gpt4': {
        'requests_per_minute': 3500,
        'delay_between_requests': 0.02
    }
    # ... other models
}
```

### Batch Size
Modify generation batch size in the script:
```python
notes = await self.generate_batch(model, batch_size=50)  # Default: 1000
```

## 🚨 Troubleshooting

### Common Issues

**"API key not found"**
- Ensure `.env` file exists with all three API keys
- Check that keys are valid and have sufficient credits
- Verify key names match exactly: `OPENAI_API_KEY`, `GOOGLE_API_KEY`, `XAI_API_KEY`

**"Rate limit exceeded"**
- Script includes automatic rate limiting
- If errors persist, increase delays in `RATE_LIMITS` configuration
- Check API usage limits on provider dashboards

**"Incomplete notes generated"**
- Review prompt templates for clarity
- Check model-specific temperature settings
- Verify required sections are properly specified

**"Only 1-2 models working"**
- Check API keys for non-working models
- Verify account status and billing for each provider
- Review logs for model-specific error messages

### Getting Help
1. Run `python test_setup.py` to diagnose setup issues
2. Check log files in `logs/` directory for detailed error messages
3. Review generated metadata files for model-specific quality metrics

## 📝 Research Design

This tool generates data for **comparative bias analysis** across multiple AI models:

### Hypothesis Testing
- **Between-model bias differences**: Do models show different bias patterns?
- **Demographic interactions**: How do biases manifest across different patient groups?
- **Pain management disparities**: Are there systematic differences in pain treatment recommendations?

### Data Structure
- **Systematic demographics**: Controlled distributions across all models
- **Identical prompts**: Same inputs for fair model comparison
- **Standardized validation**: Consistent bias detection across outputs

### Statistical Power
- **3,000 total notes**: Sufficient sample size for robust statistical analysis
- **1,000 per model**: Balanced comparison groups
- **Multiple demographics**: Factorial design for interaction effects

## 🔜 Next Steps

After generating synthetic notes, proceed to:
1. **2_Bias_Analysis**: Compare bias patterns across models
2. **3_Statistical_Analysis**: Perform statistical tests on differences
3. **4_Visualization_Reports**: Create comparative visualizations and reports

---

**Research Ethics**: This tool generates synthetic data for bias research only. All content is fictional and not for clinical use. 