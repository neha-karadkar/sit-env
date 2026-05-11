
import os
import logging
from dotenv import load_dotenv

# Load .env file FIRST before any os.getenv() calls
load_dotenv()

class Config:
    _kv_secrets = {}

    # Key Vault secret mapping (platform reference, agent-relevant only)
    KEY_VAULT_SECRET_MAP = [
        # LLM API keys
        ("AZURE_OPENAI_API_KEY", "openai-secrets.gpt-4.1"),
        ("AZURE_OPENAI_API_KEY", "openai-secrets.azure-key"),
        ("OPENAI_API_KEY", "aba-openai-secret.openai_api_key"),
        ("ANTHROPIC_API_KEY", "anthropic-secrets.anthropic_api_key"),
        ("GOOGLE_API_KEY", "google-secrets.google_api_key"),
        # Azure Content Safety
        ("AZURE_CONTENT_SAFETY_ENDPOINT", "azure-content-safety-secrets.azure_content_safety_endpoint"),
        ("AZURE_CONTENT_SAFETY_KEY", "azure-content-safety-secrets.azure_content_safety_key"),
        # Observability DB
        ("OBS_AZURE_SQL_SERVER", "agentops-secrets.obs_sql_endpoint"),
        ("OBS_AZURE_SQL_DATABASE", "agentops-secrets.obs_azure_sql_database"),
        ("OBS_AZURE_SQL_PORT", "agentops-secrets.obs_port"),
        ("OBS_AZURE_SQL_USERNAME", "agentops-secrets.obs_sql_username"),
        ("OBS_AZURE_SQL_PASSWORD", "agentops-secrets.obs_sql_password"),
        ("OBS_AZURE_SQL_SCHEMA", "agentops-secrets.obs_azure_sql_schema"),
    ]

    # Sets for LLM kwargs logic
    _MAX_TOKENS_UNSUPPORTED = {
        "gpt-5", "gpt-5-mini", "gpt-5-nano", "gpt-5.1-chat",
        "o1", "o1-mini", "o1-preview", "o3", "o3-mini", "o3-pro", "o4-mini"
    }
    _TEMPERATURE_UNSUPPORTED = {
        "gpt-5", "gpt-5-mini", "gpt-5-nano", "gpt-5.1-chat",
        "o1", "o1-mini", "o1-preview", "o3", "o3-mini", "o3-pro", "o4-mini"
    }

    @classmethod
    def _load_keyvault_secrets(cls):
        """Load secrets from Azure Key Vault if enabled."""
        if not getattr(cls, "USE_KEY_VAULT", False):
            return {}
        if not getattr(cls, "KEY_VAULT_URI", ""):
            return {}
        try:
            AZURE_USE_DEFAULT_CREDENTIAL = getattr(cls, "AZURE_USE_DEFAULT_CREDENTIAL", False)
            if AZURE_USE_DEFAULT_CREDENTIAL:
                from azure.identity import DefaultAzureCredential
                credential = DefaultAzureCredential()
            else:
                from azure.identity import ClientSecretCredential
                tenant_id = os.getenv("AZURE_TENANT_ID", "")
                client_id = os.getenv("AZURE_CLIENT_ID", "")
                client_secret = os.getenv("AZURE_CLIENT_SECRET", "")
                if not (tenant_id and client_id and client_secret):
                    logging.warning("Service Principal credentials incomplete. Key Vault access will fail.")
                    return {}
                credential = ClientSecretCredential(
                    tenant_id=tenant_id,
                    client_id=client_id,
                    client_secret=client_secret
                )
            from azure.keyvault.secrets import SecretClient
            client = SecretClient(vault_url=cls.KEY_VAULT_URI, credential=credential)
            # Group refs by secret name
            from collections import defaultdict
            import json
            refs_by_secret = defaultdict(list)
            for attr, ref in getattr(cls, "KEY_VAULT_SECRET_MAP", []):
                if "." in ref:
                    secret_name, json_key = ref.split(".", 1)
                else:
                    secret_name, json_key = ref, None
                refs_by_secret[secret_name].append((attr, json_key))
            kv_secrets = {}
            for secret_name, refs in refs_by_secret.items():
                try:
                    secret = client.get_secret(secret_name)
                    if not secret or not secret.value:
                        logging.debug(f"Key Vault: secret '{secret_name}' is empty or missing")
                        continue
                    raw_value = secret.value.lstrip('\ufeff')
                    has_json_key = any(json_key is not None for _, json_key in refs)
                    if has_json_key:
                        try:
                            data = json.loads(raw_value)
                        except Exception:
                            logging.debug(f"Key Vault: secret '{secret_name}' could not be parsed as JSON")
                            continue
                        if not isinstance(data, dict):
                            logging.debug(f"Key Vault: secret '{secret_name}' value is not a JSON object")
                            continue
                        for attr, json_key in refs:
                            if json_key is not None:
                                val = data.get(json_key)
                                if attr in kv_secrets:
                                    continue
                                if val is not None and val != "":
                                    kv_secrets[attr] = str(val)
                                else:
                                    logging.debug(f"Key Vault: key '{json_key}' not found in secret '{secret_name}' (field {attr})")
                    else:
                        for attr, json_key in refs:
                            if json_key is None and raw_value:
                                kv_secrets[attr] = raw_value
                                break
                except Exception as exc:
                    logging.debug(f"Key Vault: failed to fetch secret '{secret_name}': {exc}")
                    continue
            cls._kv_secrets = kv_secrets
            return kv_secrets
        except Exception as exc:
            logging.warning(f"Key Vault: failed to load secrets: {exc}")
            return {}

    @classmethod
    def _validate_api_keys(cls):
        provider = (getattr(cls, "MODEL_PROVIDER", "") or "").lower()
        if provider == "openai":
            if not getattr(cls, "OPENAI_API_KEY", ""):
                raise ValueError("OPENAI_API_KEY is required for OpenAI provider")
        elif provider == "azure":
            if not getattr(cls, "AZURE_OPENAI_API_KEY", ""):
                raise ValueError("AZURE_OPENAI_API_KEY is required for Azure OpenAI provider")
            if not getattr(cls, "AZURE_OPENAI_ENDPOINT", ""):
                raise ValueError("AZURE_OPENAI_ENDPOINT is required for Azure OpenAI provider")
        elif provider == "anthropic":
            if not getattr(cls, "ANTHROPIC_API_KEY", ""):
                raise ValueError("ANTHROPIC_API_KEY is required for Anthropic provider")
        elif provider == "google":
            if not getattr(cls, "GOOGLE_API_KEY", ""):
                raise ValueError("GOOGLE_API_KEY is required for Google provider")

    @classmethod
    def get_llm_kwargs(cls):
        kwargs = {}
        model_lower = (getattr(cls, "LLM_MODEL", "") or "").lower()
        if not any(model_lower.startswith(m) for m in cls._TEMPERATURE_UNSUPPORTED):
            kwargs["temperature"] = getattr(cls, "LLM_TEMPERATURE", None)
        if any(model_lower.startswith(m) for m in cls._MAX_TOKENS_UNSUPPORTED):
            kwargs["max_completion_tokens"] = getattr(cls, "LLM_MAX_TOKENS", None)
        else:
            kwargs["max_tokens"] = getattr(cls, "LLM_MAX_TOKENS", None)
        return kwargs

    @classmethod
    def validate(cls):
        cls._validate_api_keys()

def _initialize_config():
    # Load Key Vault config from .env
    USE_KEY_VAULT = os.getenv("USE_KEY_VAULT", "").lower() in ("true", "1", "yes")
    KEY_VAULT_URI = os.getenv("KEY_VAULT_URI", "")
    AZURE_USE_DEFAULT_CREDENTIAL = os.getenv("AZURE_USE_DEFAULT_CREDENTIAL", "").lower() in ("true", "1", "yes")

    Config.USE_KEY_VAULT = USE_KEY_VAULT
    Config.KEY_VAULT_URI = KEY_VAULT_URI
    Config.AZURE_USE_DEFAULT_CREDENTIAL = AZURE_USE_DEFAULT_CREDENTIAL

    # Load Key Vault secrets if enabled
    if USE_KEY_VAULT:
        Config._load_keyvault_secrets()

    # Azure AI Search variables (not used in this agent, skip)

    # Service Principal variables (only if not using DefaultAzureCredential)
    AZURE_SP_VARS = ["AZURE_TENANT_ID", "AZURE_CLIENT_ID", "AZURE_CLIENT_SECRET"]
    if not AZURE_USE_DEFAULT_CREDENTIAL:
        for var in AZURE_SP_VARS:
            value = None
            if USE_KEY_VAULT and var in Config._kv_secrets:
                value = Config._kv_secrets[var]
            else:
                value = os.getenv(var)
            if not value:
                logging.warning(f"Configuration variable {var} not found in .env file")
                value = ""
            setattr(Config, var, value)

    # All other config variables (priority: Key Vault > .env > warn)
    CONFIG_VARIABLES = [
        # General
        "ENVIRONMENT",
        # LLM / Model
        "MODEL_PROVIDER",
        "LLM_MODEL",
        "LLM_TEMPERATURE",
        "LLM_MAX_TOKENS",
        # LLM API keys
        "AZURE_OPENAI_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GOOGLE_API_KEY",
        "AZURE_OPENAI_ENDPOINT",
        # Content Safety
        "AZURE_CONTENT_SAFETY_ENDPOINT",
        "AZURE_CONTENT_SAFETY_KEY",
        "CONTENT_SAFETY_ENABLED",
        "CONTENT_SAFETY_SEVERITY_THRESHOLD",
        # Agent identity
        "AGENT_NAME",
        "AGENT_ID",
        "PROJECT_NAME",
        "PROJECT_ID",
        "SERVICE_NAME",
        "SERVICE_VERSION",
        # Observability DB
        "OBS_DATABASE_TYPE",
        "OBS_AZURE_SQL_SERVER",
        "OBS_AZURE_SQL_DATABASE",
        "OBS_AZURE_SQL_PORT",
        "OBS_AZURE_SQL_USERNAME",
        "OBS_AZURE_SQL_PASSWORD",
        "OBS_AZURE_SQL_SCHEMA",
        "OBS_AZURE_SQL_TRUST_SERVER_CERTIFICATE",
        # Domain-specific
        "LLM_MODELS",
        "VALIDATION_CONFIG_PATH",
        "VERSION",
    ]

    for var in CONFIG_VARIABLES:
        # Special handling for Service Principal vars (already loaded above)
        if var in AZURE_SP_VARS and AZURE_USE_DEFAULT_CREDENTIAL:
            continue
        value = None
        if USE_KEY_VAULT and var in Config._kv_secrets:
            value = Config._kv_secrets[var]
        else:
            value = os.getenv(var)
        # Type conversions
        if not value:
            # OBS_AZURE_SQL_TRUST_SERVER_CERTIFICATE defaults to "yes" if not found
            if var == "OBS_AZURE_SQL_TRUST_SERVER_CERTIFICATE":
                value = "yes"
            else:
                logging.warning(f"Configuration variable {var} not found in .env file")
                value = "" if var not in ("LLM_TEMPERATURE", "LLM_MAX_TOKENS", "OBS_AZURE_SQL_PORT") else None
        # Numeric conversions
        if value and var == "LLM_TEMPERATURE":
            try:
                value = float(value)
            except ValueError:
                logging.warning(f"Invalid float value for {var}: {value}")
                value = None
        elif value and var in ("LLM_MAX_TOKENS", "OBS_AZURE_SQL_PORT"):
            try:
                value = int(value)
            except ValueError:
                logging.warning(f"Invalid integer value for {var}: {value}")
                value = None
        # LLM_MODELS: parse as JSON if present
        if var == "LLM_MODELS" and value:
            import json
            try:
                value = json.loads(value)
            except Exception:
                logging.warning(f"Invalid JSON for LLM_MODELS: {value}")
                value = []
        setattr(Config, var, value)

# Initialize config at module import
_initialize_config()

# Settings instance (backward compatibility with observability module)
settings = Config()
