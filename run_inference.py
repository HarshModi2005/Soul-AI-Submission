"""
Inference module for Named Entity Recognition (NER) system.

This module provides functionality for running inference on text using trained NER models.
It includes functions for entity extraction, prediction, and result formatting.
"""

import os
import json
import time
import torch
from transformers import BertTokenizerFast, BertForTokenClassification
import argparse
from typing import List, Dict, Any, Set
import numpy as np

def load_model(model_path):
    """Load the BERT NER model from the specified path"""
    # Ensure model_path is absolute
    if not os.path.isabs(model_path):
        # Get the directory where the script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, model_path)
    
    print(f"Loading model from {model_path}...")
    
    # Check if path exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path not found: {model_path}")
    
    # Load tokenizer and model
    try:
        # Use BertTokenizerFast instead of BertTokenizer for offset mapping support
        tokenizer = BertTokenizerFast.from_pretrained(model_path)
        model = BertForTokenClassification.from_pretrained(model_path)
        model.eval()
    except Exception as e:
        raise RuntimeError(f"Error loading model: {e}")
    
    # Load tag mappings
    tag_mappings_path = os.path.join(model_path, "tag_mappings.json")
    if os.path.exists(tag_mappings_path):
        with open(tag_mappings_path, "r") as f:
            tag_mappings = json.load(f)
            id_to_tag = {int(k): v for k, v in tag_mappings["id_to_tag"].items()}
    elif hasattr(model.config, "id2label"):
        id_to_tag = model.config.id2label
    else:
        # Fallback: create basic mapping
        num_labels = model.config.num_labels
        id_to_tag = {i: f"TAG_{i}" for i in range(num_labels)}
    
    print(f"Model loaded successfully with {len(id_to_tag)} entity types")
    return tokenizer, model, id_to_tag

def predict_entities(text, tokenizer, model, id_to_tag):
    """
    Predicts named entities in the provided text using the trained model.
    
    Args:
        text (str): Input text to process
        tokenizer: Tokenizer to use for preparing the input
        model: Trained NER model
        id_to_tag (Dict): Mapping from prediction IDs to entity tags
    
    Returns:
        List[Dict]: List of detected entities with their positions and labels
    """
    # Preprocessing time measurement
    start_time = time.time()
    
    # Tokenize the input text and prepare for model
    inputs = tokenizer(text, return_tensors="pt", return_offsets_mapping=True, padding=True, truncation=True)
    offset_mapping = inputs.pop("offset_mapping")
    
    # Get model predictions
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Process predictions
    predictions = outputs.logits.argmax(-1).squeeze().tolist()
    
    # Initialize entities list and current entity tracking
    entities = []
    current_entity = None
    
    # Process each token and its prediction
    for i, (pred, offset) in enumerate(zip(predictions, offset_mapping.squeeze().tolist())):
        # Skip special tokens
        if offset[0] == offset[1]:
            continue
        
        # Get predicted tag
        tag = id_to_tag.get(pred, "O")
        
        # Handle entity boundaries
        if tag.startswith("B-"):
            # Save previous entity if exists
            if current_entity:
                entity_text = text[current_entity["start"]:current_entity["end"]]
                if entity_text.strip():
                    entities.append({
                        "text": entity_text,
                        "start": current_entity["start"],
                        "end": current_entity["end"],
                        "label": current_entity["type"]
                    })
            
            # Start new entity
            entity_type = tag[2:] if tag.startswith("B-") else tag
            current_entity = {
                "type": entity_type,
                "start": offset[0],
                "end": offset[1]
            }
        
        # Handle last entity if exists
        if current_entity:
            entity_text = text[current_entity["start"]:current_entity["end"]]
            if entity_text.strip():
                entities.append({
                    "text": entity_text,
                    "start": current_entity["start"],
                    "end": current_entity["end"],
                    "label": current_entity["type"]
                })
        
        # Post-process: clean up entity text and fix boundaries if needed
        processed_entities = []
        for entity in entities:
            entity_text = entity["text"].strip()
            if entity_text:
                # Find actual position in original text
                entity_start = text.find(entity_text, max(0, entity["start"] - 5))
                if entity_start != -1:
                    processed_entities.append({
                        "text": entity_text,
                        "start": entity_start,
                        "end": entity_start + len(entity_text),
                        "label": entity["label"]
                    })
                else:
                    # Fallback to original positions
                    processed_entities.append({
                        "text": entity_text,
                        "start": entity["start"],
                        "end": entity["start"] + len(entity_text),
                        "label": entity["label"]
                    })
        
        # Map numeric labels to human-readable labels if needed
        for entity in processed_entities:
            if entity["label"].startswith("LABEL_"):
                label_num = entity["label"].split("_")[1]
                # Map to entity type based on observed patterns in your CSV
                if label_num == "0":
                    entity["label"] = "LOCATION"
                elif label_num == "2":
                    entity["label"] = "ORGANIZATION"
                elif label_num == "3":
                    entity["label"] = "PERSON"
                elif label_num == "4":
                    entity["label"] = "LOCATION"  # Tower is typically a location
                elif label_num == "6":
                    entity["label"] = "ORGANIZATION"  # Inc is part of org
                elif label_num == "7":
                    entity["label"] = "PERSON"  # Last names
    
    return processed_entities

def fallback_predict_entities(text, tokenizer, model, id_to_tag, non_entity_labels={"O", "LABEL_8"}):
    """
    Alternative entity prediction function with a more robust approach.
    Used as a fallback when the primary prediction method fails.
    
    Args:
        text (str): Input text to process
        tokenizer: Tokenizer to use for preparing the input
        model: Trained NER model
        id_to_tag (Dict): Mapping from prediction IDs to entity tags
        non_entity_labels (Set[str]): Set of labels indicating non-entities
    
    Returns:
        List[Dict]: List of detected entities with their positions and labels
    """
    # Tokenize the text
    tokens = text.split()
    word_level_predictions = []
    
    # Process text in chunks for better handling of long inputs
    max_length = tokenizer.model_max_length
    for i in range(0, len(tokens), max_length):
        chunk_tokens = tokens[i:i+max_length]
        chunk_text = " ".join(chunk_tokens)
        
        # Tokenize and get model predictions
        inputs = tokenizer(chunk_text, return_tensors="pt", return_offsets_mapping=True)
        token_offsets = inputs.pop("offset_mapping").squeeze()
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        predictions = outputs.logits.argmax(-1).squeeze().tolist()
        
        # Map subword tokens back to original words
        for j, (pred, offset) in enumerate(zip(predictions, token_offsets)):
            # Skip special tokens
            if offset[0] == offset[1]:
                continue
                
            # Find which word this token is part of
            word_idx = -1
            curr_len = 0
            for k, word in enumerate(chunk_tokens):
                curr_len += len(word) + 1  # +1 for space
                if offset[0] < curr_len:
                    word_idx = k + i  # adjust for global word index
                    break
            
            if word_idx != -1:
                predicted_tag = id_to_tag.get(pred, "O")
                # Only add if not in non_entity_labels
                if predicted_tag not in non_entity_labels:
                    word_level_predictions.append((word_idx, predicted_tag))
    
    # Extract entities
    entities = []
    i = 0
    
    while i < len(word_level_predictions):
        word_idx, tag = word_level_predictions[i]
        
        # Skip non-entity tags
        if tag in non_entity_labels:
            i += 1
            continue
            
        if tag.startswith("B-") or (tag != "O" and not tag.startswith("I-")):
            entity_type = tag[2:] if tag.startswith("B-") else tag
            start_idx = word_idx
            end_idx = word_idx + 1
            
            # Find the end of this entity
            j = i + 1
            while j < len(word_level_predictions):
                next_idx, next_tag = word_level_predictions[j]
                
                if ((next_tag.startswith("I-") and next_tag[2:] == entity_type) or 
                    (next_tag == entity_type)):
                    end_idx = next_idx + 1
                    j += 1
                else:
                    break
            
            # Get the entity text
            if start_idx < len(tokens) and end_idx <= len(tokens):
                entity_tokens = tokens[start_idx:end_idx]
                entity_text = " ".join(entity_tokens)
                
                # Find position in original text
                start_pos = text.find(entity_text)
                if start_pos != -1:
                    # Map numeric labels to standard entity types
                    if entity_type.startswith("LABEL_"):
                        label_num = entity_type.split("_")[1]
                        # Convert numeric labels to human-readable entity types
                        if label_num == "0":
                            entity_type = "LOCATION"
                        elif label_num == "2":
                            entity_type = "ORGANIZATION"
                        elif label_num == "3":
                            entity_type = "PERSON"
                        elif label_num == "4":
                            entity_type = "LOCATION"
                        elif label_num == "6":
                            entity_type = "ORGANIZATION"
                        elif label_num == "7":
                            entity_type = "PERSON"
                        
                        # Store entity with its position and type
                        entities.append({
                            "text": entity_text,
                            "start": start_pos,
                            "end": start_pos + len(entity_text),
                            "label": entity_type
                        })
                
                i = j
            else:
                i += 1
    
    return entities

def display_entities(entities, text):
    """
    Display the detected entities in a human-readable format.
    
    This function presents entities both as a list and visually highlighted
    within the original text context.
    
    Args:
        entities (list): List of entity dictionaries with text, positions, and labels.
        text (str): Original text where entities were detected.
    """
    if not entities:
        print("No entities found.")
        return
    
    # Display numbered list of entities with details
    print("\nDetected entities:")
    for i, entity in enumerate(entities):
        print(f"{i+1}. \"{entity['text']}\" - {entity['label']} (position: {entity['start']}-{entity['end']})")
    
    # Create visual representation with entity highlighting
    print("\nText with highlighted entities:")
    last_end = 0
    highlighted_text = ""
    
    # Sort entities by start position to process them in order
    for entity in sorted(entities, key=lambda x: x['start']):
        # Add text before this entity
        highlighted_text += text[last_end:entity['start']]
        # Add the entity with highlighting format
        highlighted_text += f"[{entity['text']}]({entity['label']})"
        last_end = entity['end']
    
    # Add any remaining text after the last entity
    highlighted_text += text[last_end:]
    print(highlighted_text)

def main():
    parser = argparse.ArgumentParser(description='Run inference on BERT NER model')
    parser.add_argument('--model_path', type=str, default='data/models/bert_ner_model',
                        help='Path to the model directory')
    parser.add_argument('--text', type=str, 
                        help='Text to analyze for named entities')
    parser.add_argument('--interactive', action='store_true',
                        help='Run in interactive mode')
    
    args = parser.parse_args()
    
    # Load the model
    tokenizer, model, id_to_tag = load_model(args.model_path)
    
    if args.interactive:
        print("\nEntering interactive mode. Type 'exit' to quit.")
        while True:
            text = input("\nEnter text to analyze: ")
            if text.lower() == 'exit':
                break
            
            entities = predict_entities(text, tokenizer, model, id_to_tag)
            display_entities(entities, text)
    
    elif args.text:
        entities = predict_entities(args.text, tokenizer, model, id_to_tag)
        display_entities(entities, args.text)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()