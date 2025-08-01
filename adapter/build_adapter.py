from .basic_adapter import BasicAdapter
from .aug_adapter import AugAdapter
from .selectors import *
from cfg import args
import os


def build_adapter_from_cfg():
    if args.selector_type == 'optimal':
        selector = OptimalSelector(os.path.join(args.dump_dir, f'{args.dataset.lower()}_choices{args.postfix}.json'))
    elif args.selector_type == 'llm':
        selector = LLMSelectorFromJson(
            f'./data/llm_output/{args.dataset.lower()}_output{args.postfix}.json', 
            os.path.join(args.dump_dir, f'{args.dataset.lower()}_choices{args.postfix}.json'), 
            os.path.join(args.dump_dir, f'{args.dataset.lower()}_choices_r2{args.postfix}.json')
        )
    elif args.selector_type == 'llm_r2':
        selector = LLMSelectorFromJson(
            f'./data/llm_output/{args.dataset.lower()}_output_r2{args.postfix}.json', 
            os.path.join(args.dump_dir, f'{args.dataset.lower()}_choices_r2{args.postfix}.json'), 
            os.path.join(args.dump_dir, f'{args.dataset.lower()}_choices_r3{args.postfix}.json'),
        )
    else:
        raise ValueError('Selector not supported')
    
    if args.adapter_type == 'basic':
        return BasicAdapter(selector)
    elif args.adapter_type == 'aug':
        return AugAdapter(selector)
    else:
        raise ValueError('Adapter not supported')
