"""Implement an ArgParser common to both brew_poison.py and dist_brew_poison.py ."""

import argparse

def options():
    """Construct the central argument parser, filled with useful defaults.

    The first block is essential to test poisoning in different scenarios.
    The options following afterwards change the algorithm in various ways and are set to reasonable defaults.
    """
    parser = argparse.ArgumentParser(description='Construct poisoned training data for the given network and dataset')

    
    ###########################################################################
    parser.add_argument('--f')
    # Central:
    parser.add_argument('--net', default='ResNet50', type=lambda s: [str(item) for item in s.split(',')])
    parser.add_argument('--dataset', default='Facial_recognition_extended', type=str)
    parser.add_argument('--recipe', default='gradient-matching', type=str, choices=['gradient-matching', 'gradient-matching-private', 'mttp',
                                                                                    'hidden-trigger', 'hidden-trigger-mt', 'gradient-matching-mt', 'feature-matching',
                                                                                    'patch', 'gradient-matching-hidden', 'meta', 'meta-v2', 'meta-v3', 'meta-first-order', 'naive', 'dirty-label', 'label-consistent'])
                                                                                    
    parser.add_argument('--threatmodel', default='clean-single-source', type=str, choices=['clean-single-source', 'clean-multi-source', 'clean-all-source', 'third-party', 'self-betrayal', 'all-to-all'])
    parser.add_argument('--num_source_classes', default=1, type=int, help='Number of source classes (for many-to-one attacks)')
    parser.add_argument('--scenario', default='finetuning', type=str, choices=['from-scratch', 'transfer', 'finetuning'])

    # Reproducibility management:
    parser.add_argument('--poisonkey', default='3-1', type=str, help='Initialize poison setup with this key.')  # Take input such as 05-1 for [0, 5] as the sources and 1 as the target
    parser.add_argument('--system_seed', default=None, type=int, help='Initialize the system with this key.')
    parser.add_argument('--poison_seed', default=None, type=int, help='Initialize the poisons with this key.')
    parser.add_argument('--model_seed', default=123456, type=int, help='Initialize the model with this key.')
    parser.add_argument('--deterministic', action='store_true', help='Disable CUDNN non-determinism.')

    # Files and folders
    parser.add_argument('--name', default='', type=str, help='Name tag for the result table and possibly for export folders.')
    parser.add_argument('--poison_path', default='poisons/', type=str)
    parser.add_argument('--model_savepath', default='models/', type=str)
    ###########################################################################

    # Mixing defense
    parser.add_argument('--mixing_method', default=None, type=str, help='Which mixing data augmentation to use.')
    parser.add_argument('--mixing_disable_correction', action='store_false', help='Disable correcting the loss term appropriately after data mixing.')
    parser.add_argument('--mixing_strength', default=None, type=float, help='How strong is the mixing.')
    parser.add_argument('--disable_adaptive_attack', action='store_false', help='Do not use a defended model as input for poisoning. [Defend only in poison validation]')
    parser.add_argument('--defend_features_only', action='store_true', help='Only defend during the initial pretraining before poisoning. [Defend only in pretraining]')
    # Note: If --disable_adaptive_attack and --defend_features_only, then the defense is never activated


    # Privacy defenses
    parser.add_argument('--gradient_noise', default=None, type=float, help='Add custom gradient noise during training.')
    parser.add_argument('--gradient_clip', default=None, type=float, help='Add custom gradient clip during training.')

    # Adversarial defenses
    parser.add_argument('--defense_type', default=None, type=str, help='Add custom novel defenses.')
    parser.add_argument('--defense_strength', default=None, type=float, help='Add custom strength to novel defenses.')
    parser.add_argument('--defense_steps', default=None, type=int, help='Override default number of adversarial steps taken by the defense.')
    parser.add_argument('--defense_sources', default=None, type=str, help='Different choices for source selection. Options: shuffle/sep-half/sep-1/sep-10')
    
    # Adaptive attack variants
    parser.add_argument('--padversarial', default=None, type=str, help='Use adversarial steps during poison brewing.')
    parser.add_argument('--pmix', action='store_true', help='Use mixing during poison brewing [Uses the mixing specified in mixing_type].')

    # Poison brewing:
    parser.add_argument('--attackoptim', default='signAdam', type=str)
    parser.add_argument('--attackiter', default=250, type=int)
    parser.add_argument('--init', default='rand', type=str)  # randn / rand
    parser.add_argument('--tau', default=0.1, type=float)
    parser.add_argument('--scheduling', action='store_false', help='Disable step size decay.')
    parser.add_argument('--poison_scheduler', default='cosine', type=str, help='Scheduler for poison learning rate.')
    parser.add_argument('--source_criterion', default='cross-entropy', type=str, help='Loss criterion for poison loss')
    parser.add_argument('--restarts', default=1, type=int, help='How often to restart the attack.')
    
    # MTTP params
    parser.add_argument('--bkd_epochs', default=1, type=int)
    parser.add_argument('--bkd_batch_size', default=128, type=int, help='Batch size for backdoor training')
    parser.add_argument('--bkd_lr', default=0.001, type=float, help='Learning rate for backdoor training')
    parser.add_argument('--bkd_training_mode', default='full_data', type=str, choices=['full_data', 'poison_only'], help='Mode of backdoor training.')
    parser.add_argument('--mtt_loss', default='MSE', type=str, choices=['MSE', 'similarity'], help='Loss function for MTTP')
    parser.add_argument('--mtt_validate_every', default=None, type=int, help='How often to validate the model during MTTP training. If None, no validation is performed.')
    
    # Feature Matching params
    parser.add_argument('--sample_from_trajectory', default=False, action='store_true', help='Whether to sample embedding space from training trajectory')
    parser.add_argument('--sample_every', default=5, type=int, help='How often to sample from the trajectory')
    parser.add_argument('--sample_same_idx', default=False, action='store_true', help='For ensemble models, whether to sample the same index from the trajectory for all models')
    parser.add_argument('--dist_reg_weight', default=None, type=float, help="Weight for Distribution Regularizer")
    
    # Poisoning
    parser.add_argument('--paugment', action='store_true', help='Augment poison batch during optimization')
    parser.add_argument('--pbatch', default=64, type=int, help='Poison batch size during optimization')
    parser.add_argument('--data_aug', type=str, default='default', help='Mode of diff. data augmentation.')

    # Poisoning algorithm changes
    parser.add_argument('--full_data', action='store_true', help='Use full train data for poisoning (instead of just the poison images)')
    parser.add_argument('--ensemble', default=1, type=int, help='Ensemble of networks to brew the poison on')
    parser.add_argument('--sample_gradient', action='store_true', help='Sample the gradient of a network instead of averaging gradients (for ensemble models)')
    parser.add_argument('--stagger', default=None, type=str, help='Stagger the network ensemble if it exists', choices=['firstn', 'full', 'inbetween'])
    parser.add_argument('--step', action='store_true', help='Optimize the model for one epoch.')
    parser.add_argument('--step_every', default=5, type=int, help='How often to step the model during poisoning.')
    parser.add_argument('--step_on_poison', default=False, action='store_true', help='Step the model on poisoned data only.')
    parser.add_argument('--validate_every', default=10, type=int, help='How often to to validate the model during training.')
    parser.add_argument('--train_max_epoch', default=40, type=int, help='Train only up to this epoch before poisoning.')
    parser.add_argument('--clean_training_only', default=False, action='store_true', help='Only train the clean data')

    # Use only a subset of the dataset:
    parser.add_argument('--ablation', default=1.0, type=float, help='What percent of data (including poisons) to use for validation')

    # Gradient Matching - Specific Options
    parser.add_argument('--loss', default='similarity', type=str)  # similarity is stronger in  difficult situations

    # These are additional regularization terms for gradient matching. We do not use them, but it is possible
    # that scenarios exist in which additional regularization of the poisoned data is useful.
    parser.add_argument('--centreg', default=0, type=float)
    parser.add_argument('--normreg', default=0, type=float)
    parser.add_argument('--repel', default=0, type=float)
    parser.add_argument('--visreg', default=None, type=str)
    parser.add_argument('--vis_weight', default=1, type=float)
    parser.add_argument('--scale', default=1.0, type=float)
    
    # Specific Options for a metalearning recipe
    parser.add_argument('--nadapt', default=1, type=int, help='Meta unrolling steps')

    # Validation behavior
    parser.add_argument('--vruns', default=3, type=int, help='How often to re-initialize and check source after retraining')
    parser.add_argument('--vnet', default=None, type=lambda s: [str(item) for item in s.split(',')], help='Evaluate poison on this victim model. Defaults to --net')
    parser.add_argument('--retrain_from_init', action='store_true', help='Additionally evaluate by retraining on the same model initialization.')
    parser.add_argument('--skip_clean_training', action='store_true', help='Skip clean training. This is only suggested for attacks that do not depend on a clean model.')

    # Optimization setup
    parser.add_argument('--optimization', default='conservative-sgd', type=str, help='Optimization Strategy')
    
    # Strategy overrides:
    parser.add_argument('--batch_size', default=64, type=int, help='Override default batch_size of --optimization strategy')
    parser.add_argument('--lr', default=0.1, type=float, help='Override default learning rate of --optimization strategy')
    parser.add_argument('--augment', action='store_true', default=False, help='Use data augmentation during training.')

    # Optionally, datasets can be stored within RAM:
    parser.add_argument('--cache_dataset', action='store_true', help='Cache the entire thing :>')

    # Debugging:
    parser.add_argument('--dryrun', default=False, action='store_true', help='This command runs every loop only a single time.')
    

    parser.add_argument('--train_from_scratch', default=False, action='store_true', help='Train model from scratch')
    parser.add_argument('--save_poison', default=None, help='Export poisons into a given format. Options are full/limited/numpy.')
    parser.add_argument('--save_clean_model', default=False, action='store_true', help='Save the clean model train on specific seed')
    parser.add_argument('--save_backdoored_model', default=False, action='store_true', help='Save the backdoored model train on specific seed')
    parser.add_argument('--exp_name', default=None, help='Save experimental results to a separate folder')

    # Backdoor attack:
    parser.add_argument('--keep_sources', action='store_true', default=True, help='Do we keep the sources are used for testing attack success rate?')
    parser.add_argument('--sources_train_rate', default=1.0, type=float, help='Fraction of source_class trainset that can be selected crafting poisons')
    parser.add_argument('--sources_selection_rate', default=1.0, type=float, help='Fraction of sources to be selected for calculating source_grad in gradient-matching')
    parser.add_argument('--source_gradient_batch', default=64, type=int, help='Batch size for sources train gradient computing')
    parser.add_argument('--val_max_epoch', default=40, type=int, help='Train only up to this epoch for final validation.')
    parser.add_argument('--retrain_max_epoch', default=20, type=int, help='Train only up to this epoch for retraining during crafting.')
    parser.add_argument('--retrain_scenario', default=None, type=str, choices=['from-scratch', 'finetuning', 'transfer'], help='Scenario for retraining and evaluating on the poisoned dataset')
    parser.add_argument('--retrain_reinit_seed', default=False, action='store_true', help="Reinit seed for retraining")
    parser.add_argument('--trigger', default='sunglasses', type=str, help='Trigger type')
    parser.add_argument('--digital_train', action='store_true', default=False, help='Adding digital trigger instead of physical ones during training')
    parser.add_argument('--digital_test', action='store_true', default=False, help='Adding digital trigger instead of physical ones during inference')
    parser.add_argument('--digital_trigger_path', default='digital_triggers')
    parser.add_argument('--retrain_iter', default=100, type=int, help='Start retraining every <retrain_iter> iterations')
    parser.add_argument('--source_selection_strategy', default="max_gradient", type=str, choices=['max_gradient', 'max_loss'], help='source selection strategy')
    parser.add_argument('--poison_selection_strategy', default="max_gradient", type=str, help='Poison selection strategy')
    
    # Poison properties / controlling the strength of the attack:
    parser.add_argument('--eps', default=16, type=float, help='Epsilon bound of the attack in a ||.||_p norm. p=Inf for all recipes except for "patch".')
    parser.add_argument('--alpha', default=0.1, type=float, help='Fraction of target_class training data that is poisoned by adding pertubation')

    #--------------------------------------------------------------------------------------------------------------------------"
    # Defenses
    parser.add_argument('--defense', default=None, type=str, help='Which filtering defense to use.')
    parser.add_argument('--firewall', default=None, type=float, help='How strong is the defense.')
    parser.add_argument('--inspection_path', default=None, type=str, help='Path for inspection set')
    parser.add_argument('--clean_budget', default=0.2, type=float, help='Fraction of test dataset to use for defense.')
    
    # CUDA_VISIBLE_DEVICES
    parser.add_argument("--devices", type=str, default="0,1")    
    
    # Suspicion check
    parser.add_argument("--suspicion_check", default=False, action='store_true')
    
    # BLC setting
    parser.add_argument("--random_placement", action='store_true')

    # Denoising and noising
    parser.add_argument("--denoise", action='store_true')
    parser.add_argument("--gaussian_noise", action='store_true')

    return parser
