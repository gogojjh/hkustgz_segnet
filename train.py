""" 
training code
"""
from ruamel.yaml import YAML
import wandb
import argparse
import logging
import os
import torch
from apex import amp  # mixed-float training for speed-up

from config import cfg
import datasets
import loss
import network
import optimizer
