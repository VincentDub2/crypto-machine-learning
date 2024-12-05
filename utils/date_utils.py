
from datetime import datetime

def date_to_timestamp(date_str):
    """
    Convertit une date en chaîne de caractères au format 'YYYY-MM-DD'
    en un timestamp en millisecondes.

    :param date_str: La date sous forme de chaîne (ex: '2022-01-01')
    :return: Timestamp en millisecondes
    """
    # Convertir la chaîne de caractères en objet datetime
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    # Convertir l'objet datetime en timestamp en secondes, puis en millisecondes
    timestamp_ms = int(dt.timestamp() * 1000)
    return timestamp_ms


