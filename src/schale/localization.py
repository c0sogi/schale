from typing import Literal
import locale
import os
import sys
from schale import literal

Lang = Literal["kr", "jp", "cn", "en"]

PYROXENE_NAMES: dict[Lang, str] = {
    "en": "Pyroxene",
    "kr": "청휘석",
    "jp": "青輝石",
    "cn": "青輝石",
}

CREDIT_NAMES: dict[Lang, str] = {
    "en": "Credit",
    "kr": "크레딧",
    "jp": "クレジット",
    "cn": "學園幣",
}
BOUNTY_TICKET_NAMES: dict[Lang, str] = {
    "en": "Bounty Ticket",
    "kr": "현상수배 티켓",
    "jp": "指名手配チケット",
    "cn": "懸賞通緝票券",
}
SCRIMMAGE_TICKET_NAMES: dict[Lang, str] = {
    "en": "Scrimmage Ticket",
    "kr": "학원교류회 티켓",
    "jp": "学園交流会チケット",
    "cn": "學園交流會票券",
}

BOUNTY_TRANSLATIONS: dict[literal.BountyStageType, dict[Lang, str]] = {
    "ChaserA": {
        "en": "Overpass",
        "kr": "고가도로",
        "jp": "ハイウェイ",
        "cn": "高架公路",
    },
    "ChaserB": {
        "en": "Abandoned Train",
        "kr": "버려진 기차",
        "jp": "捨てられた列車",
        "cn": "被遺棄的列車",
    },
    "ChaserC": {
        "en": "Besieged Classroom",
        "kr": "습격 받은 교실",
        "jp": "襲撃された校舎",
        "cn": "被襲擊的教室",
    },
}

WEEK_DUNGEON_TRANSLATIONS: dict[literal.WeekDungeonStageType, dict[Lang, str]] = {
    "FindGift": {
        "en": "Item Retrieval",
        "kr": "슬럼피아 광장",
        "jp": "スランピア広場",
        "cn": "史林皮亞廣場",
    },
    "Blood": {
        "en": "Base Defense",
        "kr": "폐허의 군수공장",
        "jp": "軍需工場の廃墟",
        "cn": "廢墟軍需工廠",
    },
}


SCHOOL_DUNGEON_TRANSLATIONS: dict[literal.SchoolDungeonStageType, dict[Lang, str]] = {
    "SchoolA": {
        "en": "Trinity",
        "kr": "트리니티",
        "jp": "トリニティ",
        "cn": "三一",
    },
    "SchoolB": {
        "en": "Gehenna",
        "kr": "게헨나",
        "jp": "ゲヘナ",
        "cn": "格黑娜",
    },
    "SchoolC": {
        "en": "Millennium",
        "kr": "밀레니엄",
        "jp": "ミレニアム",
        "cn": "千年",
    },
}

CATEGORY_TRANSLATIONS: dict[literal.StageCategory, dict[Lang, str]] = {
    "Campaign": {
        "en": "Campaign",
        "kr": "캠페인",
        "jp": "任務",
        "cn": "任務",
    },
    "Bounty": {
        "en": "Bounty",
        "kr": "현상수배",
        "jp": "指名手配",
        "cn": "懸賞通緝",
    },
    "WeekDungeon": {
        "en": "Commissions",
        "kr": "특별의뢰",
        "jp": "特別依頼",
        "cn": "特別委托",
    },
    "SchoolDungeon": {
        "en": "Scrimmage",
        "kr": "학원교류회",
        "jp": "学園交流会",
        "cn": "學園交流會",
    },
}
ITEM_TRANSLATIONS: dict[literal.ItemCategory, dict[Lang, str]] = {
    "Material": {
        "en": "Material",
        "kr": "재료",
        "jp": "素材",
        "cn": "材料",
    },
    "Coin": {
        "en": "Currency",
        "kr": "재화",
        "jp": "アイテム",
        "cn": "道具",
    },
    "CharacterExpGrowth": {
        "en": "Character EXP",
        "kr": "캐릭터 경험치",
        "jp": "キャラクターEXP",
        "cn": "經驗書",
    },
    "Favor": {
        "en": "Gift",
        "kr": "선물",
        "jp": "贈り物",
        "cn": "禮物",
    },
    "SecretStone": {
        "en": "Eleph",
        "kr": "엘레프",
        "jp": "神名文字",
        "cn": "神名文字",
    },
    "Consumable": {
        "en": "Consumable",
        "kr": "소모품",
        "jp": "消費アイテム",
        "cn": "消耗品",
    },
    "Collectible": {
        "en": "Collectible",
        "kr": "수집품",
        "jp": "収集品",
        "cn": "收藏品",
    },
}


EQUIPMENT_TRANSLATIONS: dict[literal.EquipmentCategory, dict[Lang, str]] = {
    "Exp": {
        "en": "Enhancement Stone",
        "kr": "강화석",
        "jp": "強化珠",
        "cn": "強化珠",
    },
    "WeaponExpGrowthA": {
        "en": "Weapon Part (Spring)",
        "kr": "무기 부품 (스프링)",
        "jp": "武器パーツ (スプリング)",
        "cn": "武器零件 (擊錘)",
    },
    "WeaponExpGrowthB": {
        "en": "Weapon Part (Hammer)",
        "kr": "무기 부품 (해머)",
        "jp": "武器パーツ (ハンマー)",
        "cn": "武器零件 (彈簧)",
    },
    "WeaponExpGrowthC": {
        "en": "Weapon Part (Barrel)",
        "kr": "무기 부품 (총열)",
        "jp": "武器パーツ (銃身)",
        "cn": "武器零件 (槍管)",
    },
    "WeaponExpGrowthZ": {
        "en": "Weapon Part (Firing Pin)",
        "kr": "무기 부품 (공이)",
        "jp": "武器パーツ (撃針)",
        "cn": "武器零件 (擊針)",
    },
    "Hat": {
        "en": "Hat",
        "kr": "모자",
        "jp": "帽子",
        "cn": "帽子",
    },
    "Gloves": {
        "en": "Gloves",
        "kr": "장갑",
        "jp": "グローブ",
        "cn": "手套",
    },
    "Shoes": {
        "en": "Shoes",
        "kr": "신발",
        "jp": "シューズ",
        "cn": "鞋子",
    },
    "Bag": {
        "en": "Bag",
        "kr": "가방",
        "jp": "バッグ",
        "cn": "包包",
    },
    "Badge": {
        "en": "Badge",
        "kr": "배지",
        "jp": "バッジ",
        "cn": "徽章",
    },
    "Hairpin": {
        "en": "Hairpin",
        "kr": "헤어핀",
        "jp": "ヘアピン",
        "cn": "髮夾",
    },
    "Charm": {
        "en": "Amulet",
        "kr": "부적",
        "jp": "お守り",
        "cn": "護符",
    },
    "Necklace": {
        "en": "Necklace",
        "kr": "목걸이",
        "jp": "ネックレス",
        "cn": "項鍊",
    },
    "Watch": {
        "en": "Wristwatch",
        "kr": "손목시계",
        "jp": "腕時計",
        "cn": "手錶",
    },
}


def detect_user_lang() -> Lang:
    def _normalize_locale_name(val: str) -> Lang | None:
        """
        Normalize locale strings from:
        - POSIX: ko_KR, ja_JP, zh_CN
        - Windows: Korea_Korean, Japanese_Japan, Chinese_China
        """
        s = val.lower()

        # POSIX-style locales
        if s.startswith("ko_"):
            return "kr"
        if s.startswith("ja_"):
            return "jp"
        if s.startswith("zh_"):
            return "cn"

        # Windows locale names
        if s.startswith("korea"):
            return "kr"
        if s.startswith("japan"):
            return "jp"
        if s.startswith("chinese"):
            return "cn"

        return None

    # 1. locale.getlocale() (non-deprecated)
    try:
        lang, _ = locale.getlocale()
        if lang:
            code = _normalize_locale_name(lang)
            if code:
                return code
    except Exception:
        pass

    # 2. Environment variables
    for key in ("LC_ALL", "LANG", "LC_CTYPE"):
        val = os.environ.get(key)
        if not val:
            continue

        # strip encoding part: ko_KR.UTF-8 -> ko_KR
        base = val.split(".", 1)[0]
        code = _normalize_locale_name(base)
        if code:
            return code

    # 3. Windows codepage fallback
    if sys.platform.startswith("win"):
        try:
            enc = locale.getpreferredencoding(False).lower()
            if "949" in enc:
                return "kr"  # Korean
            if "932" in enc:
                return "jp"  # Japanese
            if "936" in enc or "950" in enc:
                return "cn"  # Chinese (simplified / traditional)
        except Exception:
            pass

    # 4. Ultimate fallback
    return "en"


USER_LANG = detect_user_lang()
ITEM_METADATA_URL = f"https://schaledb.com/data/{USER_LANG}/items.min.json"
EQUIPMENT_METADATA_URL = f"https://schaledb.com/data/{USER_LANG}/equipment.min.json"
FURNITURE_METADATA_URL = f"https://schaledb.com/data/{USER_LANG}/furniture.min.json"
GROUPS_URL = "https://schaledb.com/data/groups.min.json"
STAGES_URL = f"https://schaledb.com/data/{USER_LANG}/stages.min.json"
STUDENTS_URL = f"https://schaledb.com/data/{USER_LANG}/students.min.json"
