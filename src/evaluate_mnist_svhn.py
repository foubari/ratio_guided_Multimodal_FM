"""
Evaluate coherence of generated MNIST-SVHN pairs.

Metrics:
1. Coherence accuracy: P(label(MNIST) == label(SVHN))
2. Marginal quality: Accuracy on MNIST and SVHN separately

Usage:
    python src/evaluate_mnist_svhn.py --guidance_methods none mc_feng --guidance_strengths 0.0 0.5 1.0 --num_samples 500
"""
import argparse
import torch
import json
import os
import sys
from tqdm import tqdm

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.svhn_classifier import SVHNClassifier, MNISTClassifier32
from src.models.unet_flexible import FlowMatchingUNetMNIST, FlowMatchingUNetSVHN
from src.models.ratio_flexible import RatioEstimatorMNISTSVHN
from src.sample_mnist_svhn import sample_bimodal_guided_mnist_svhn
from src.utils import set_seed, load_checkpoint


def evaluate_coherence(samples_mnist, samples_svhn, mnist_classifier, svhn_classifier, device):
    """
    Compute coherence metrics.

    Args:
        samples_mnist: [N, 1, 32, 32]
        samples_svhn: [N, 3, 32, 32]
        mnist_classifier: MNISTClassifier32
        svhn_classifier: SVHNClassifier
        device: device

    Returns:
        dict with coherence_acc, pred_mnist, pred_svhn
    """
    mnist_classifier.eval()
    svhn_classifier.eval()

    with torch.no_grad():
        logits_mnist = mnist_classifier(samples_mnist.to(device))
        logits_svhn = svhn_classifier(samples_svhn.to(device))

    pred_mnist = logits_mnist.argmax(dim=1).cpu().numpy()
    pred_svhn = logits_svhn.argmax(dim=1).cpu().numpy()

    coherence_acc = (pred_mnist == pred_svhn).mean()

    return {
        'coherence_acc': float(coherence_acc),
        'num_samples': len(samples_mnist)
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate MNIST-SVHN guided sampling')
    parser.add_argument('--guidance_methods', nargs='+', default=['none', 'mc_feng'],
                        help='Guidance methods to evaluate')
    parser.add_argument('--guidance_strengths', nargs='+', type=float, default=[0.0, 0.5, 1.0],
                        help='Guidance strengths')
    parser.add_argument('--mc_batch_size', type=int, default=256,
                        help='Number of MC samples')
    parser.add_argument('--loss_type', type=str, default='disc',
                        help='Loss type for ratio estimator')
    parser.add_argument('--num_samples', type=int, default=500,
                        help='Number of samples per configuration')
    parser.add_argument('--num_steps', type=int, default=100,
                        help='ODE integration steps')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    args = parser.parse_args()

    set_seed(args.seed)
    print(f"Random seed: {args.seed}")

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load classifiers
    print("Loading classifiers...")

    mnist_classifier = MNISTClassifier32().to(device)
    svhn_classifier = SVHNClassifier().to(device)

    mnist_clf_path = 'checkpoints/mnist32_classifier.pth'
    svhn_clf_path = 'checkpoints/svhn_classifier.pth'

    if not os.path.exists(mnist_clf_path):
        print(f"ERROR: MNIST classifier not found: {mnist_clf_path}")
        print("Please train first: python src/train_classifiers_mnist_svhn.py")
        return

    if not os.path.exists(svhn_clf_path):
        print(f"ERROR: SVHN classifier not found: {svhn_clf_path}")
        print("Please train first: python src/train_classifiers_mnist_svhn.py")
        return

    mnist_classifier.load_state_dict(torch.load(mnist_clf_path, map_location=device))
    svhn_classifier.load_state_dict(torch.load(svhn_clf_path, map_location=device))
    print(f"  Loaded MNIST classifier from: {mnist_clf_path}")
    print(f"  Loaded SVHN classifier from: {svhn_clf_path}")

    # Load FM models
    print("Loading FM models...")

    fm_mnist = FlowMatchingUNetMNIST(img_size=32).to(device)
    fm_svhn = FlowMatchingUNetSVHN().to(device)

    path_mnist = 'checkpoints/flow_mnist32_best.pth'
    path_svhn = 'checkpoints/flow_svhn_best.pth'

    if not os.path.exists(path_mnist) or not os.path.exists(path_svhn):
        print(f"ERROR: FM checkpoints not found")
        return

    load_checkpoint(fm_mnist, path_mnist, device)
    load_checkpoint(fm_svhn, path_svhn, device)
    print(f"  Loaded FM_mnist and FM_svhn")

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
                ratio_estimator = RatioEstimatorMNISTSVHN(loss_type=args.loss_type).to(device)
                path_ratio = f'checkpoints/ratio_{args.loss_type}_mnist_svhn_best.pth'

                if not os.path.exists(path_ratio):
                    print(f"ERROR: Ratio estimator not found: {path_ratio}")
                    continue

                ratio_estimator.load_state_dict(torch.load(path_ratio, map_location=device))
                print(f"  Loaded ratio estimator")

            # Sample
            print(f"  Sampling {args.num_samples} pairs...")
            samples_mnist, samples_svhn = sample_bimodal_guided_mnist_svhn(
                fm_mnist=fm_mnist,
                fm_svhn=fm_svhn,
                ratio_estimator=ratio_estimator,
                guidance_method=method,
                guidance_strength=strength,
                num_samples=args.num_samples,
                num_steps=args.num_steps,
                device=device,
                mc_batch_size=args.mc_batch_size
            )

            # Evaluate
            print(f"  Evaluating coherence...")
            metrics = evaluate_coherence(
                samples_mnist, samples_svhn,
                mnist_classifier, svhn_classifier, device
            )

            result = {
                'method': method,
                'guidance_strength': strength,
                'experiment': 'mnist_svhn',
                **metrics
            }

            results.append(result)

            print(f"  → Coherence accuracy: {metrics['coherence_acc']:.3f}")

    # Save results
    os.makedirs('outputs/mnist_svhn', exist_ok=True)
    output_path = 'outputs/mnist_svhn/evaluation_results.json'
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
