"""
Text prompt processing utilities
"""
import re
import string
from typing import List, Tuple, Dict


class PromptProcessor:
    """Process and normalize text prompts for object detection"""
    
    def __init__(self):
        self.stop_words = {'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being'}
        
    def clean_prompt(self, prompt: str) -> str:
        """Clean and normalize a text prompt"""
        # Remove extra whitespace
        prompt = re.sub(r'\s+', ' ', prompt.strip())
        
        # Remove special characters except commas and periods
        prompt = re.sub(r'[^\w\s,.]', '', prompt)
        
        # Convert to lowercase
        prompt = prompt.lower()
        
        return prompt
    
    def split_objects(self, prompt: str) -> List[str]:
        """
        Split prompt into individual object terms
        
        Examples:
            "person, car, dog" -> ["person", "car", "dog"]
            "red car and blue truck" -> ["red car", "blue truck"]
        """
        prompt = self.clean_prompt(prompt)
        
        # Split by common separators
        objects = []
        
        # First try comma separation
        if ',' in prompt:
            objects = [obj.strip() for obj in prompt.split(',')]
        # Try 'and' separation  
        elif ' and ' in prompt:
            objects = [obj.strip() for obj in prompt.split(' and ')]
        # Single object
        else:
            objects = [prompt.strip()]
        
        # Filter empty strings
        objects = [obj for obj in objects if obj and len(obj.strip()) > 0]
        
        return objects
    
    def expand_synonyms(self, object_term: str) -> List[str]:
        """
        Expand object term with common synonyms
        
        This helps with detection recall by providing alternative terms
        """
        synonyms_map = {
            'person': ['person', 'human', 'people', 'man', 'woman', 'individual'],
            'car': ['car', 'vehicle', 'automobile', 'sedan', 'suv'],
            'truck': ['truck', 'lorry', 'pickup', 'semi'],
            'bike': ['bike', 'bicycle', 'cycle'],
            'motorcycle': ['motorcycle', 'motorbike', 'bike'],
            'dog': ['dog', 'canine', 'puppy'],
            'cat': ['cat', 'feline', 'kitten'],
            'bird': ['bird', 'avian'],
            'phone': ['phone', 'cellphone', 'mobile', 'smartphone'],
            'laptop': ['laptop', 'computer', 'notebook'],
            'bag': ['bag', 'purse', 'backpack', 'handbag'],
            'chair': ['chair', 'seat'],
            'table': ['table', 'desk'],
            'bottle': ['bottle', 'container'],
            'cup': ['cup', 'mug', 'glass'],
        }
        
        object_term = object_term.lower().strip()
        
        # Check for exact match
        if object_term in synonyms_map:
            return synonyms_map[object_term]
        
        # Check for partial matches
        for key, synonyms in synonyms_map.items():
            if object_term in synonyms:
                return synonyms_map[key]
            if any(syn in object_term for syn in synonyms):
                return synonyms_map[key]
        
        # Return original if no synonyms found
        return [object_term]
    
    def create_grounding_prompt(self, objects: List[str]) -> str:
        """
        Create formatted prompt for Grounding DINO
        
        Grounding DINO expects: "object1 . object2 . object3"
        """
        if not objects:
            return ""
        
        # Clean each object term
        cleaned_objects = [self.clean_prompt(obj) for obj in objects]
        
        # Join with periods and spaces
        grounding_prompt = " . ".join(cleaned_objects)
        
        # Ensure it ends with a period
        if not grounding_prompt.endswith('.'):
            grounding_prompt += " ."
        
        return grounding_prompt
    
    def parse_detection_prompt(self, prompt: str, use_synonyms: bool = True) -> Tuple[str, List[str]]:
        """
        Parse user prompt into Grounding DINO format and object list
        
        Args:
            prompt: Raw user input like "find cars and people"
            use_synonyms: Whether to expand with synonyms
            
        Returns:
            grounding_prompt: Formatted for Grounding DINO
            object_list: List of individual objects
        """
        # Extract objects from prompt
        objects = self.split_objects(prompt)
        
        # Expand with synonyms if requested
        if use_synonyms:
            expanded_objects = []
            for obj in objects:
                synonyms = self.expand_synonyms(obj)
                # Use the most common/generic term
                expanded_objects.append(synonyms[0])
            objects = expanded_objects
        
        # Create grounding prompt
        grounding_prompt = self.create_grounding_prompt(objects)
        
        return grounding_prompt, objects
    
    def validate_prompt(self, prompt: str) -> Tuple[bool, str]:
        """
        Validate if prompt is suitable for object detection
        
        Returns:
            is_valid: Whether prompt is valid
            message: Validation message
        """
        if not prompt or not prompt.strip():
            return False, "Prompt cannot be empty"
        
        if len(prompt.strip()) > 200:
            return False, "Prompt too long (max 200 characters)"
        
        # Check for valid characters
        if not re.match(r'^[a-zA-Z0-9\s,.\-_]+$', prompt):
            return False, "Prompt contains invalid characters"
        
        # Parse objects
        objects = self.split_objects(prompt)
        if not objects:
            return False, "No valid objects found in prompt"
        
        if len(objects) > 10:
            return False, "Too many objects (max 10)"
        
        return True, "Valid prompt"


# Global instance for easy use
prompt_processor = PromptProcessor()