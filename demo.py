import polars as pl
from typing import Tuple
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from src.comic_extraction import (
    video_name_to_comic_name, 
    video_names_to_comic_names,
    video_name_to_comic_name_with_stats, 
    video_names_to_comic_names_with_stats
)
from src.prompts import (
    example_prompt,
    french_context_prompt,
    few_shot_prompt,
    baseline_prompt
)
from src.prompt_tester import PromptTester
from src.evaluation import extract_expected_name_from_labels

console = Console()
df = pl.read_csv("data/france_inter.csv", n_rows=100)


def test_prompt_variations():
    """Teste différentes variations de prompts."""
    console.print(Panel.fit("[bold magenta]Test des variations de prompts[/bold magenta]", border_style="magenta"))
    
    tester = PromptTester(dataset=df)
    
    # Test tous les prompts sur un échantillon
    prompts_to_test = [
        (baseline_prompt, "Baseline"),
        (example_prompt, "With System Message"),
        (french_context_prompt, "French Context"),
        (few_shot_prompt, "Few-Shot Examples")
    ]
    
    for prompt_func, prompt_name in prompts_to_test:
        console.print(f"\n[bold]Testing: {prompt_name}[/bold]")
        tester.test_prompt(prompt_func, prompt_name, sample_size=10)
    
    # Afficher les résultats avec accuracy
    tester.display_metrics()
    tester.compare_prompts()
    
    # Montrer quelques exemples pour le meilleur prompt
    if tester.accuracy_metrics:
        best_prompt = max(tester.accuracy_metrics.items(), key=lambda x: x[1]['exact_accuracy'])[0]
        console.print(f"\n[bold]Exemples pour le meilleur prompt ({best_prompt}):[/bold]")
        tester.show_evaluation_samples(best_prompt, n_samples=10)
    
    # Sauvegarder tous les résultats
    for prompt_func, prompt_name in prompts_to_test:
        filename = f"{prompt_name.lower().replace(' ', '_')}_results.csv"
        tester.save_results(prompt_name, filename)


if __name__ == "__main__":
    console.print("[bold blue]Demo Extraction Comiques France Inter[/bold blue]\n")
    
    # Test fonction 1 - structured output
    console.print("[bold green]Test Fonction 1: video_name -> comic_name (structured output)[/bold green]")
    test_video = df['video_name'][0]
    result1 = video_name_to_comic_name(test_video)
    console.print(f"Video: {test_video}")
    console.print(f"Resultat: '{result1}'")
    
    # Test fonction 2 - CSV format  
    console.print("\n[bold green]Test Fonction 2: video_names -> comic_names (CSV format)[/bold green]")
    test_videos = df['video_name'][:5].to_list()
    results2 = video_names_to_comic_names(test_videos)
    
    table = Table(title="Resultats batch CSV")
    table.add_column("Video", style="yellow", max_width=50)
    table.add_column("Comique", style="cyan")
    
    for i, video in enumerate(test_videos):
        comic = results2[i] if i < len(results2) else "ERROR"
        table.add_row(video, f"'{comic}'")
    
    console.print(table)
    
    # Test avec stats
    console.print("\n[bold green]Test avec statistiques (tokens/temps/cout)[/bold green]")
    comic_name, stats = video_name_to_comic_name_with_stats(test_video)
    console.print(f"Input tokens: {stats['prompt_tokens']}")
    console.print(f"Output tokens: {stats['completion_tokens']}")
    console.print(f"Temps: {stats['duration']:.3f}s")
    console.print(f"Cout: ${stats['cost_usd']:.6f}")
    
    console.print("\n" + "="*60 + "\n")
    
    # Tester différentes approches
    test_prompt_variations()
    
    console.print("\n[bold green]Demo terminee ! Consultez les fichiers CSV generes.[/bold green]")