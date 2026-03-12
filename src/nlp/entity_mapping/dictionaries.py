from __future__ import annotations

ISSUER_ALIASES: dict[str, list[str]] = {
    "SBER": ["сбер", "сбербанк", "sber", "sberbank", "пао сбербанк"],
    "GAZP": ["газпром", "gazprom", "пао газпром"],
    "LKOH": ["лукойл", "lukoil", "пao лукойл", "пао лукойл"],
    "NVTK": ["новатэк", "novatek", "пао новатэк"],
    "ROSN": ["роснефть", "rosneft", "пао нк роснефть"],
    "YDEX": ["яндекс", "yandex", "мкпао яндекс"],
    "GMKN": ["норникель", "гмк норникель", "nornickel"],
    "VTBR": ["втб", "vtb", "банк втб"],
}

TICKER_TO_ISSUER: dict[str, str] = {
    "SBER": "Sberbank",
    "GAZP": "Gazprom",
    "LKOH": "Lukoil",
    "NVTK": "Novatek",
    "ROSN": "Rosneft",
    "YDEX": "Yandex",
    "GMKN": "Nornickel",
    "VTBR": "VTB",
}

SECTOR_KEYWORDS: dict[str, list[str]] = {
    "banking": ["банк", "банков", "кредит", "ипотек"],
    "oil_gas": ["нефт", "газ", "энергет", "топлив"],
    "technology": ["it", "технол", "интернет", "цифров"],
    "metals_mining": ["металл", "горнодоб", "никель", "сталь"],
}

SECTOR_FALLBACK_TICKER: dict[str, str] = {
    "banking": "SBER",
    "oil_gas": "GAZP",
    "technology": "YDEX",
    "metals_mining": "GMKN",
}
