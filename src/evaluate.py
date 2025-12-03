"""
Evaluate coherence of generated pairs.

Metrics:
1. Coherence accuracy: P(label(x) == label(y))
2. Marginal quality: Accuracy on x and y separately

Usage:
    python src/evaluate.py --transform_type rotate90 --guidance_methods none grad_log_ratio --guidance_strengths 0.0 1.0 2.0 --num_samples 1000
"""
import argparse
import torch
import json
import os
import sys

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.classifier import MNISTClassifier
from src.models.flow_matching import FlowMatchingModel
from src.models.ratio_estimator import RatioEstimator
from src.utils.flow_utils import sample_bimodal_guided
from src.utils.path_utils import get_checkpoint_path


def evaluate_coherence(samples_x, samples_y, classifier, device):
    """
    Compute coherence metrics.

    Args:
        samples_x, samples_y: [N, 1, 28, 28]
        classifier: MNIST classifier
        device: device

    Returns:
        dict with coherence_acc, pred_x, pred_y
    """
    classifier.eval()

    with torch.no_grad():
        logits_x = classifier(samples_x.to(device))
        logits_y = classifier(samples_y.to(device))

    pred_x = logits_x.argmax(dim=1).cpu().numpy()
    pred_y = logits_y.argmax(dim=1).cpu().numpy()

    coherence_acc = (pred_x == pred_y).mean()

    return {
        'coherence_acc': float(coherence_acc),
        'num_samples': len(samples_x)
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate guided sampling')
    parser.add_argument('--transform_type', type=str, default='rotate90',
                        help='Transformation type')
    parser.add_argument('--guidance_methods', nargs='+', default=['none', 'grad_log_ratio'],
                        help='Guidance methods to evaluate')
    parser.add_argument('--guidance_strengths', nargs='+', type=float, default=[0.0, 1.0, 2.0],
                        help='Guidance strengths (gamma)')
    parser.add_argument('--loss_type', type=str, default='disc',
                        help='Loss type for ratio estimator')
    parser.add_argument('--num_samples', type=int, default=1000,
                        help='Number of samples per configuration')
    parser.add_argument('--num_steps', type=int, default=100,
                        help='ODE integration steps')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda/cpu)')
    args = parser.parse_args()

    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load classifier
    print("Loading MNIST classifier...")
    classifier = MNISTClassifier().to(device)
    classifier_path = 'checkpoints/mnist_classifier.pth'

    if not os.path.exists(classifier_path):
        print(f"ERROR: Classifier not found: {classifier_path}")
        print("Please train classifier first: python src/train_classifier.py")
        return

    classifier.load_state_dict(torch.load(classifier_path, map_location=device))
    print(f"  Loaded from: {classifier_path}")

    # Load FM models
    print("Loading FM models...")
    fm_x = FlowMatchingModel().to(device)
    fm_y = FlowMatchingModel().to(device)

    path_x = get_checkpoint_path('flow', 'x', None, 'best')
    path_y = get_checkpoint_path('flow', 'y', args.transform_type, 'best')

    if not os.path.exists(path_x) or not os.path.exists(path_y):
        print(f"ERROR: FM checkpoints not found")
        print(f"  FM_x: {path_x}")
        print(f"  FM_y: {path_y}")
        return

    fm_x.load_state_dict(torch.load(path_x, map_location=device))
    fm_y.load_state_dict(torch.load(path_y, map_location=device))
    print(f"  Loaded FM_x and FM_y")

    # Evaluate all configurations
    results = []

    for method in args.guidance_methods:
        for strength in args.guidance_strengths:
            # Skip invalid combinations
            if method == 'none' and strength > 0:
                continue

            print(f"\n{'='*60}")
            print(f"Evaluating: method={method}, gamma={strength}")
            print(f"{'='*60}")

            # Load ratio estimator if needed
            ratio_estimator = None
            if method != 'none':
                ratio_estimator = RatioEstimator(loss_type=args.loss_type).to(device)
                path_ratio = get_checkpoint_path('ratio', args.loss_type, args.transform_type, 'best')

                if not os.path.exists(path_ratio):
                    print(f"ERROR: Ratio estimator not found: {path_ratio}")
                    continue

                ratio_estimator.load_state_dict(torch.load(path_ratio, map_location=device))
                print(f"  Loaded ratio estimator")

            # Sample
            print(f"  Sampling {args.num_samples} pairs...")
            samples_x, samples_y = sample_bimodal_guided(
                fm_x=fm_x,
                fm_y=fm_y,
                ratio_estimator=ratio_estimator,
                guidance_method=method,
                guidance_strength=strength,
                num_samples=args.num_samples,
                num_steps=args.num_steps,
                device=device
            )

            # Evaluate
            print(f"  Evaluating coherence...")
            metrics = evaluate_coherence(samples_x, samples_y, classifier, device)

            result = {
                'method': method,
                'guidance_strength': strength,
                'transform_type': args.transform_type,
                **metrics
            }

            results.append(result)

            print(f"  → Coherence accuracy: {metrics['coherence_acc']:.3f}")

    # Save results
    os.makedirs('outputs', exist_ok=True)
    output_path = 'outputs/evaluation_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*60}")

    # Print summary
    print("\nSummary:")
    for result in results:
        print(f"  {result['method']:20s} γ={result['guidance_strength']:.1f} → coherence={result['coherence_acc']:.3f}")


if __name__ == '__main__':
    main()
