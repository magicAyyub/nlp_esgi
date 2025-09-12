import openai
from pydantic import BaseModel
from typing import List, Tuple
import time

# Configuration du client 
client = openai.OpenAI(
     base_url="https://api.groq.com/openai/v1",
     api_key="",
)

# Modèle Pydantic pour structured output
class ComicName(BaseModel):
    name: str

# Configuration des prix (par million de tokens)
MODEL_PRICING = {
    "openai/gpt-oss-20b": {"input": 0.002, "output": 0.002},
    "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79}
}


def video_name_to_comic_name(video_name: str) -> str:
    """
    Fonction 1: video_name: str -> comic_name: str
    Extrait le nom d'un comique/humoriste d'un titre de vidéo France Inter.
    Utilise structured output.
    Retourne une chaîne vide si aucun comique n'est trouvé.
    """
    reply = client.chat.completions.parse(
        messages=[{
            "role": "user", 
            "content": f"Extract the comedian's name from this France Inter video title. If no comedian is found, return an empty string: '{video_name}'"
        }],
        model="openai/gpt-oss-20b",
        response_format=ComicName,
    )
    
    return reply.choices[0].message.parsed.name if reply.choices[0].message.parsed.name else ""


def video_names_to_comic_names(video_names: List[str]) -> List[str]:
    """
    Fonction 2: video_names: list[str] -> comic_names: list[str]
    Extrait les noms de comiques de plusieurs titres de vidéos.
    Utilise format CSV (sans structured output).
    """
    video_list = "\n".join([f"- {name}" for name in video_names])
    
    prompt = f"""Extract comedian names from these France Inter video titles and format your response as CSV.
Start your reply with ```csv
video_name;comic_names

Video titles:
{video_list}

For each video title, extract the comedian's name. If no comedian is found, leave the comic_names field empty.
Format: video_title;comedian_name (one line per video)
"""

    reply = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile",
    )
    
    # Parse CSV response
    response_content = reply.choices[0].message.content
    comic_names = []
    lines = response_content.split('\n')
    
    csv_started = False
    for line in lines:
        if line.strip().startswith('```csv') or line.strip() == 'video_name;comic_names':
            csv_started = True
            continue
        if csv_started and line.strip() and not line.strip().startswith('```'):
            if ';' in line:
                parts = line.strip().split(';')
                if len(parts) >= 2:
                    comic_name = parts[1].strip()
                    comic_names.append(comic_name if comic_name else "")
                else:
                    comic_names.append("")
    
    return comic_names


# Fonctions avec stats pour le PromptTester
def video_name_to_comic_name_with_stats(video_name: str) -> Tuple[str, dict]:
    """Version avec stats pour le PromptTester."""
    start_time = time.time()
    
    reply = client.chat.completions.parse(
        messages=[{
            "role": "user", 
            "content": f"Extract the comedian's name from this France Inter video title. If no comedian is found, return an empty string: '{video_name}'"
        }],
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


def video_names_to_comic_names_with_stats(video_names: List[str]) -> Tuple[List[str], dict]:
    """Version avec stats pour le PromptTester."""
    start_time = time.time()
    
    video_list = "\n".join([f"- {name}" for name in video_names])
    
    prompt = f"""Extract comedian names from these France Inter video titles and format your response as CSV.
Start your reply with ```csv
video_name;comic_names

Video titles:
{video_list}

For each video title, extract the comedian's name. If no comedian is found, leave the comic_names field empty.
Format: video_title;comedian_name (one line per video)
"""

    reply = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile",
    )
    
    duration = time.time() - start_time
    usage = reply.usage
    
    # Calcul du coût
    pricing = MODEL_PRICING["llama-3.3-70b-versatile"]
    cost = (usage.prompt_tokens / 1_000_000) * pricing["input"] + (usage.completion_tokens / 1_000_000) * pricing["output"]
    
    stats = {
        "model": "llama-3.3-70b-versatile",
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
        "duration": duration,
        "cost_usd": cost
    }
    
    # Parse CSV response
    response_content = reply.choices[0].message.content
    comic_names = []
    lines = response_content.split('\n')
    
    csv_started = False
    for line in lines:
        if line.strip().startswith('```csv') or line.strip() == 'video_name;comic_names':
            csv_started = True
            continue
        if csv_started and line.strip() and not line.strip().startswith('```'):
            if ';' in line:
                parts = line.strip().split(';')
                if len(parts) >= 2:
                    comic_name = parts[1].strip()
                    comic_names.append(comic_name if comic_name else "")
                else:
                    comic_names.append("")
    
    return comic_names, stats