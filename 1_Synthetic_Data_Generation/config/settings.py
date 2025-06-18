"""
Configuration settings for synthetic operative note generation.
"""

# =============================================================================
# DEMOGRAPHIC DISTRIBUTIONS
# =============================================================================

DEMOGRAPHICS = {
    'race': {
        'White': 0.2,
        'Black': 0.2,
        'Asian': 0.2,
        'Native American': 0.2,
        'Other': 0.2
    },
    'gender': {
        'male': 0.4,
        'female': 0.4,
        'non-binary': 0.2
    },
    'ethnicity': {
        'Hispanic': 0.5,
        'non-Hispanic': 0.5
    },
    'insurance': {
        'Medicare': 0.33,
        'Medicaid': 0.33,
        'Private Insurance': 0.34
    },
    'primary_language': {
        'English': 0.6,
        'Spanish': 0.2,
        'Other': 0.2  # Mandarin, Arabic, etc.
    }
}

# =============================================================================
# MEDICAL PARAMETERS
# =============================================================================

# Age range for patients
AGE_RANGE = (18, 80)

# Surgical procedures to generate notes for
PROCEDURES = [
    'appendectomy',
    'cholecystectomy', 
    'coronary artery bypass grafting',
    'total knee arthroplasty',
    'hysterectomy'
]

# Pain severity levels
PAIN_LEVELS = ['mild', 'moderate', 'severe']

# Common comorbidities
COMORBIDITIES = [
    'diabetes mellitus',
    'hypertension',
    'coronary artery disease',
    'chronic kidney disease',
    'asthma',
    'COPD',
    'obesity'
]

# =============================================================================
# GENERATION PARAMETERS
# =============================================================================

# Number of notes to generate per model
NOTES_PER_MODEL = 1000

# AI models for comparative analysis
MODELS = {
    'gpt4': {
        'name': 'GPT-4 Turbo',
        'enabled': True,
        'api_endpoint': 'https://api.openai.com/v1/chat/completions',
        'model_name': 'gpt-4-turbo-preview',
        'max_tokens': 4000,
        'temperature': 0.7,
        'top_p': 0.9
    },
    'gemini': {
        'name': 'Gemini Pro',
        'enabled': True,
        'api_endpoint': 'https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent',
        'model_name': 'gemini-pro',
        'max_tokens': 8000,
        'temperature': 0.7,
        'top_p': 0.95,
        'top_k': 40
    },
    'grok': {
        'name': 'Grok',
        'enabled': True,
        'api_endpoint': 'https://api.x.ai/v1/chat/completions',
        'model_name': 'grok-beta',
        'max_tokens': 4000,
        'temperature': 0.7,
        'top_p': 0.9
    }
}

# Rate limiting settings for each model
RATE_LIMITS = {
    'gpt4': {
        'requests_per_minute': 3500,
        'delay_between_requests': 0.02  # 20ms
    },
    'gemini': {
        'requests_per_minute': 60,
        'delay_between_requests': 1.0  # 1 second
    },
    'grok': {
        'requests_per_minute': 100,
        'delay_between_requests': 0.6  # 600ms
    }
}

# =============================================================================
# OUTPUT SETTINGS
# =============================================================================

# Output file paths (relative to the data generation folder)
OUTPUT_PATHS = {
    'notes': 'output/synthetic_notes_{model}_{timestamp}.json',
    'metadata': 'output/metadata_{model}_{timestamp}.json',
    'logs': 'logs/generation_{model}_{timestamp}.log'
}

# Required sections in each operative note
REQUIRED_SECTIONS = [
    'patient_demographics',
    'preoperative_assessment', 
    'procedure_details',
    'postoperative_plan',
    'pain_management'
] 