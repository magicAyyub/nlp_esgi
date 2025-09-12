"""
Module pour tester différentes approches de prompting.
Contient la classe PromptTester et les fonctions d'évaluation.
"""

import polars as pl
import time
from typing import Dict, List, Tuple, Callable
from rich.console import Console
from rich.table import Table
from rich.progress import track

from .evaluation import create_evaluation_dataset, calculate_accuracy

console = Console()

class PromptTester:
    """Classe pour tester différentes versions de prompts sur le dataset France Inter"""

    def __init__(self, dataset: pl.DataFrame = None):
        self.results = {}
        self.metrics = {}
        self.accuracy_metrics = {}
        self.dataset = dataset if dataset is not None else self.load_dataset()
        
        # Créer le dataset d'évaluation avec les noms attendus
        self.eval_dataset = create_evaluation_dataset(self.dataset)

    def load_dataset(self) -> pl.DataFrame:
        """Charge le dataset depuis un fichier CSV."""
        df = pl.read_csv("data/france_inter.csv")
        return df
    
    def test_prompt(self, prompt_func: Callable, prompt_name: str, sample_size: int = None):
        """Teste une fonction de prompt sur le dataset et stocke les résultats."""
        console.log(f"Testing prompt: {prompt_name}")
        
        # Limitation du dataset si demandé
        test_data = self.dataset.head(sample_size) if sample_size else self.dataset
        
        results = []
        total_tokens = 0
        total_time = 0.0
        total_cost = 0.0
        
        for row in track(test_data.iter_rows(named=True), total=len(test_data), description=f"Processing with {prompt_name}"):
            video_name = row['video_name']
            try:
                comic_name, stats = prompt_func(video_name)
                
                total_tokens += stats['total_tokens']
                total_time += stats['duration']
                total_cost += stats['cost_usd']
                
                results.append({
                    "video_name": video_name,
                    "comic_name": comic_name,
                    "model": stats['model'],
                    "prompt_tokens": stats['prompt_tokens'],
                    "completion_tokens": stats['completion_tokens'],
                    "total_tokens": stats['total_tokens'],
                    "duration": stats['duration'],
                    "cost_usd": stats['cost_usd']
                })
            except Exception as e:
                console.log(f"Error processing '{video_name}': {e}")
        
        self.results[prompt_name] = results
        self.metrics[prompt_name] = {
            "total_calls": len(results),
            "total_tokens": total_tokens,
            "average_tokens": total_tokens / len(results) if results else 0,
            "total_time": total_time,
            "average_time": total_time / len(results) if results else 0,
            "total_cost": total_cost,
            "average_cost": total_cost / len(results) if results else 0,
        }
        
        # Calculer l'accuracy si on a les labels
        if hasattr(self, 'eval_dataset') and len(results) > 0:
            # Extraire les prédictions et les noms attendus
            predictions = [r["comic_name"] for r in results]
            expected_names = []
            
            for r in results:
                # Trouver le nom attendu correspondant
                matching_row = self.eval_dataset.filter(pl.col("video_name") == r["video_name"])
                if len(matching_row) > 0:
                    expected_names.append(matching_row["expected_name"][0])
                else:
                    expected_names.append("")
            
            # Calculer les métriques d'accuracy
            accuracy_metrics = calculate_accuracy(predictions, expected_names)
            self.accuracy_metrics[prompt_name] = accuracy_metrics

    def display_metrics(self):
        """Affiche les métriques de chaque prompt dans un tableau."""
        table = Table(title="Prompt Performance Metrics")
        table.add_column("Prompt", style="cyan", no_wrap=True)
        table.add_column("Total Calls", justify="right")
        table.add_column("Total Tokens", justify="right")
        table.add_column("Avg Tokens/Call", justify="right")
        table.add_column("Total Time (s)", justify="right")
        table.add_column("Avg Time/Call (s)", justify="right")
        table.add_column("Total Cost ($)", justify="right", style="yellow")
        table.add_column("Avg Cost/Call ($)", justify="right", style="yellow")
        table.add_column("Accuracy", justify="right", style="bold green")
        
        for prompt_name, metrics in self.metrics.items():
            accuracy = self.accuracy_metrics.get(prompt_name, {})
            
            table.add_row(
                prompt_name,
                str(metrics["total_calls"]),
                str(metrics["total_tokens"]),
                f"{metrics['average_tokens']:.1f}",
                f"{metrics['total_time']:.2f}",
                f"{metrics['average_time']:.2f}",
                f"${metrics['total_cost']:.4f}",
                f"${metrics['average_cost']:.4f}",
                f"{accuracy.get('exact_accuracy', 0):.1%}"
            )
        
        console.print(table)
        
        # Afficher les détails d'accuracy
        if self.accuracy_metrics:
            console.print("\n[bold]Détails des métriques d'accuracy:[/bold]")
            for prompt_name, accuracy in self.accuracy_metrics.items():
                console.print(f"\n{prompt_name}:")
                console.print(f"  • Exact matches: {accuracy['exact_matches']}/{accuracy['total_samples']} ({accuracy['exact_accuracy']:.1%})")
                console.print(f"  • Detection (vide vs nom): {accuracy['detection_accuracy']:.1%}")
                console.print(f"  • Extraction de noms: {accuracy['names_correctly_extracted']}/{accuracy['names_to_extract']} ({accuracy['name_extraction_accuracy']:.1%})")

    def compare_prompts(self):
        """Compare les résultats entre différents prompts."""
        if len(self.results) < 2:
            console.log("Need at least 2 prompts to compare")
            return
            
        # Comparaison des métriques principales
        prompt_names = list(self.results.keys())
        console.print(f"\n[bold]Comparaison des prompts:[/bold]")
        
        for prompt_name in prompt_names:
            metrics = self.metrics[prompt_name]
            accuracy = self.accuracy_metrics.get(prompt_name, {})
            console.print(f"\n{prompt_name}:")
            console.print(f"  • Accuracy exacte: {accuracy.get('exact_accuracy', 0):.1%}")
            console.print(f"  • Accuracy détection: {accuracy.get('detection_accuracy', 0):.1%}")
            console.print(f"  • Avg tokens: {metrics['average_tokens']:.1f}")
            console.print(f"  • Avg time: {metrics['average_time']:.3f}s")
            console.print(f"  • Avg cost: ${metrics['average_cost']:.4f}")

    def show_evaluation_samples(self, prompt_name: str, n_samples: int = 10):
        """Affiche quelques exemples de prédictions vs attendu pour un prompt."""
        if prompt_name not in self.results:
            console.log(f"No results found for prompt: {prompt_name}")
            return
            
        results = self.results[prompt_name][:n_samples]
        
        table = Table(title=f"Exemples de prédictions - {prompt_name}")
        table.add_column("Titre vidéo", style="yellow", max_width=40)
        table.add_column("Prédit", style="cyan")
        table.add_column("Attendu", style="green")
        table.add_column("Match", justify="center")
        
        for r in results:
            # Trouver le nom attendu correspondant
            matching_row = self.eval_dataset.filter(pl.col("video_name") == r["video_name"])
            expected = matching_row["expected_name"][0] if len(matching_row) > 0 else ""
            
            predicted = r["comic_name"]
            is_match = "✓" if predicted.strip().lower() == expected.strip().lower() else "✗"
            match_style = "green" if is_match == "✓" else "red"
            
            table.add_row(
                r["video_name"],
                f"'{predicted}'",
                f"'{expected}'",
                f"[{match_style}]{is_match}[/{match_style}]"
            )
        
        console.print(table)

    def get_results(self, prompt_name: str) -> List[Dict]:
        """Retourne les résultats pour un prompt donné."""
        return self.results.get(prompt_name, [])
    
    def save_results(self, prompt_name: str, filename: str):
        """Sauvegarde les résultats d'un prompt dans un fichier CSV."""
        results = self.get_results(prompt_name)
        if not results:
            console.log(f"No results to save for prompt: {prompt_name}")
            return

        df = pl.DataFrame(results)
        df.write_csv(filename)
        console.log(f"Results saved to {filename}")

    def create_evaluation_dataset(self, output_file: str = "evaluation_dataset.csv"):
        """
        Crée un dataset d'évaluation manuel pour mesurer l'accuracy.
        """
        console.print("[bold]Création du dataset d'évaluation[/bold]")
        console.print("Vous allez devoir annoter manuellement quelques exemples...")
        
        # Prendre un échantillon du dataset
        sample = self.dataset.head(10)
        
        evaluation_data = []
        for row in sample.iter_rows(named=True):
            video_name = row['video_name']
            console.print(f"\nTitre: [yellow]{video_name}[/yellow]")
            expected_answer = input("Nom du comique attendu (ou ENTER si aucun): ").strip()
            
            evaluation_data.append({
                "video_name": video_name,
                "expected_comic": expected_answer if expected_answer else ""
            })
        
        df_eval = pl.DataFrame(evaluation_data)
        df_eval.write_csv(output_file)
        console.log(f"Dataset d'évaluation sauvé dans {output_file}")
        
        return df_eval
