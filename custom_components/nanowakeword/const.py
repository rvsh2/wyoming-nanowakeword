"""Constants for the NanoWakeWord integration."""

DOMAIN = "nanowakeword"

CONF_TOKEN = "token"
CONF_MODEL_FILE = "model_file"
CONF_BACKUP_FILE = "backup_file"
CONF_BACKUP = "backup"
CONF_FILENAME = "filename"
CONF_RECORDING = "recording"
CONF_MODEL = "model"
CONF_VERIFY_URL = "verify_url"
CONF_VERIFY_TOKEN = "verify_token"
CONF_VERIFY_MODEL = "verify_model"

DEFAULT_PORT = 10401

BACKUP_DIR = "nanowakeword"

ATTR_ENTRY_ID = "entry_id"
ATTR_PATH = "path"

SIGNAL_DETECTION = "nanowakeword_detection_{}"
SIGNAL_BACKUP = "nanowakeword_backup_{}"

SERVICE_BACKUP = "backup"
SERVICE_RESTORE = "restore"
SERVICE_UPLOAD_MODEL = "upload_model"
SERVICE_DELETE_MODEL = "delete_model"
SERVICE_RELOAD_MODELS = "reload_models"
