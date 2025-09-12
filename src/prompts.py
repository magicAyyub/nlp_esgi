"""
Module contenant différentes variantes de prompts pour tester les approches.
Chaque fonction de prompt retourne (comic_name: str, stats: dict)
"""

import time
from typing import Tuple
from src.comic_extraction import client, ComicName, MODEL_PRICING


def example_prompt(video_name: str) -> Tuple[str, dict]:
    """
    Prompt d'exemple avec système message.
    """
    start_time = time.time()
    
    reply = client.chat.completions.parse(
        messages=[
            {"role": "system", "content": "You are a helpful assistant that extracts comedian names from video titles."},
            {"role": "user", "content": f"Extract the comedian's name from this video title: '{video_name}'"}
        ],
        model="openai/gpt-oss-20b",
        response_format=ComicName,
    )
    
    duration = time.time() - start_time
    usage = reply.usage
    
    # Calcul du coût
    pricing = MODEL_PRICING["openai/gpt-oss-20b"]
    cost = (usage.prompt_tokens / 1_000_000) * pricing["input"] + (usage.completion_tokens / 1_000_000) * pricing["output"]
    
    stats = {
        "model": "openai/gpt-oss-20b",
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
        "duration": duration,
        "cost_usd": cost
    }
    
    comic_name = reply.choices[0].message.parsed.name if reply.choices[0].message.parsed.name else ""
    return comic_name, stats


def french_context_prompt(video_name: str) -> Tuple[str, dict]:
    """
    Prompt avec contexte français/France Inter.
    """
    start_time = time.time()
    
    reply = client.chat.completions.parse(
        messages=[
            {"role": "system", "content": "Tu es un expert des émissions de France Inter. Tu connais tous les humoristes français."},
            {"role": "user", "content": f"Extrait le nom du comique/humoriste de ce titre d'émission France Inter: '{video_name}'"}
        ],
        model="openai/gpt-oss-20b",
        response_format=ComicName,
    )
    
    duration = time.time() - start_time
    usage = reply.usage
    
    pricing = MODEL_PRICING["openai/gpt-oss-20b"]
    cost = (usage.prompt_tokens / 1_000_000) * pricing["input"] + (usage.completion_tokens / 1_000_000) * pricing["output"]
    
    stats = {
        "model": "openai/gpt-oss-20b",
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
        "duration": duration,
        "cost_usd": cost
    }
    
    comic_name = reply.choices[0].message.parsed.name if reply.choices[0].message.parsed.name else ""
    return comic_name, stats


def few_shot_prompt(video_name: str) -> Tuple[str, dict]:
    """
    Prompt avec exemples (few-shot learning).
    """
    start_time = time.time()
    
    examples = """
Exemples:
- "La chronique de Thomas VDB" → Thomas VDB
- "Guillaume Meurice présente" → Guillaume Meurice  
- "Les chroniques de Charline Vanhoenacker" → Charline Vanhoenacker
- "Journal de 8h" → (aucun comique)
"""
    
    reply = client.chat.completions.parse(
        messages=[
            {"role": "user", "content": f"{examples}\n\nExtrait le nom du comique de: '{video_name}'"}
        ],
        model="openai/gpt-oss-20b",
        response_format=ComicName,
    )
    
    duration = time.time() - start_time
    usage = reply.usage
    
    pricing = MODEL_PRICING["openai/gpt-oss-20b"]
    cost = (usage.prompt_tokens / 1_000_000) * pricing["input"] + (usage.completion_tokens / 1_000_000) * pricing["output"]
    
    stats = {
        "model": "openai/gpt-oss-20b",
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
        "duration": duration,
        "cost_usd": cost
    }
    
    comic_name = reply.choices[0].message.parsed.name if reply.choices[0].message.parsed.name else ""
    return comic_name, stats


def baseline_prompt(video_name: str) -> Tuple[str, dict]:
    """
    Prompt de base minimal (notre implémentation originale).
    """
    return video_name_to_comic_name_with_stats(video_name)


# Import nécessaire pour baseline_prompt
from src.comic_extraction import video_name_to_comic_name_with_stats
