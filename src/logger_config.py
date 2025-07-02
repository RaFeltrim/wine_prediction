import logging
import sys

# Configuração básica do logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('wine_prediction.log', mode='a', encoding='utf-8')
    ]
)

logger = logging.getLogger('wine_prediction')
