"""
Unit tests for prompt processing utilities
"""
import pytest
from src.prompts import PromptProcessor


class TestPromptProcessor:
    """Test prompt processing functionality"""
    
    def setup_method(self):
        """Setup for each test"""
        self.processor = PromptProcessor()
    
    def test_clean_prompt_basic(self):
        """Test basic prompt cleaning"""
        assert self.processor.clean_prompt("  Hello World  ") == "hello world"
        assert self.processor.clean_prompt("UPPER case") == "upper case"
        assert self.processor.clean_prompt("multiple   spaces") == "multiple spaces"
    
    def test_clean_prompt_special_chars(self):
        """Test cleaning with special characters"""
        assert self.processor.clean_prompt("hello@world!") == "hello world"
        assert self.processor.clean_prompt("test, with. punctuation") == "test, with. punctuation"
        assert self.processor.clean_prompt("test#$%^&*()test") == "test test"
    
    def test_split_objects_comma_separated(self):
        """Test splitting comma-separated objects"""
        objects = self.processor.split_objects("person, car, dog")
        assert objects == ["person", "car", "dog"]
        
        objects = self.processor.split_objects("  person  ,  car  ,  dog  ")
        assert objects == ["person", "car", "dog"]
    
    def test_split_objects_and_separated(self):
        """Test splitting 'and'-separated objects"""
        objects = self.processor.split_objects("person and car and dog")
        assert objects == ["person", "car", "dog"]
        
        objects = self.processor.split_objects("red car and blue truck")
        assert objects == ["red car", "blue truck"]
    
    def test_split_objects_single(self):
        """Test splitting single object"""
        objects = self.processor.split_objects("person")
        assert objects == ["person"]
        
        objects = self.processor.split_objects("construction worker")
        assert objects == ["construction worker"]
    
    def test_split_objects_empty(self):
        """Test splitting empty/invalid prompts"""
        assert self.processor.split_objects("") == []
        assert self.processor.split_objects("   ") == []
        assert self.processor.split_objects(",,,") == []
    
    def test_expand_synonyms_known_objects(self):
        """Test synonym expansion for known objects"""
        synonyms = self.processor.expand_synonyms("car")
        assert "car" in synonyms
        assert "vehicle" in synonyms
        assert len(synonyms) > 1
        
        synonyms = self.processor.expand_synonyms("person")
        assert "person" in synonyms
        assert "human" in synonyms
    
    def test_expand_synonyms_unknown_objects(self):
        """Test synonym expansion for unknown objects"""
        synonyms = self.processor.expand_synonyms("unknown_object")
        assert synonyms == ["unknown_object"]
    
    def test_expand_synonyms_partial_match(self):
        """Test synonym expansion with partial matches"""
        synonyms = self.processor.expand_synonyms("smartphone")
        # Should match "phone" synonyms
        assert "phone" in synonyms or "smartphone" in synonyms
    
    def test_create_grounding_prompt(self):
        """Test creating Grounding DINO format prompts"""
        prompt = self.processor.create_grounding_prompt(["person", "car"])
        assert prompt == "person . car ."
        
        prompt = self.processor.create_grounding_prompt(["single"])
        assert prompt == "single ."
        
        prompt = self.processor.create_grounding_prompt([])
        assert prompt == ""
    
    def test_parse_detection_prompt_basic(self):
        """Test parsing basic detection prompts"""
        grounding_prompt, objects = self.processor.parse_detection_prompt("person, car")
        
        assert "person" in grounding_prompt
        assert "car" in grounding_prompt
        assert grounding_prompt.endswith(".")
        assert objects == ["person", "car"]
    
    def test_parse_detection_prompt_with_synonyms(self):
        """Test parsing with synonym expansion"""
        grounding_prompt, objects = self.processor.parse_detection_prompt(
            "car", use_synonyms=True
        )
        
        # Should use the primary synonym (usually "car" itself)
        assert "car" in grounding_prompt.lower()
        assert len(objects) == 1
    
    def test_parse_detection_prompt_without_synonyms(self):
        """Test parsing without synonym expansion"""
        grounding_prompt, objects = self.processor.parse_detection_prompt(
            "automobile", use_synonyms=False
        )
        
        assert "automobile" in grounding_prompt
        assert objects == ["automobile"]
    
    def test_validate_prompt_valid(self):
        """Test validation of valid prompts"""
        is_valid, message = self.processor.validate_prompt("person, car")
        assert is_valid
        assert "valid" in message.lower()
        
        is_valid, message = self.processor.validate_prompt("construction worker with helmet")
        assert is_valid
    
    def test_validate_prompt_empty(self):
        """Test validation of empty prompts"""
        is_valid, message = self.processor.validate_prompt("")
        assert not is_valid
        assert "empty" in message.lower()
        
        is_valid, message = self.processor.validate_prompt("   ")
        assert not is_valid
    
    def test_validate_prompt_too_long(self):
        """Test validation of overly long prompts"""
        long_prompt = "a" * 250  # Exceeds 200 character limit
        is_valid, message = self.processor.validate_prompt(long_prompt)
        assert not is_valid
        assert "too long" in message.lower()
    
    def test_validate_prompt_invalid_chars(self):
        """Test validation of prompts with invalid characters"""
        is_valid, message = self.processor.validate_prompt("person@#$%^&*()")
        assert not is_valid
        assert "invalid characters" in message.lower()
    
    def test_validate_prompt_too_many_objects(self):
        """Test validation of prompts with too many objects"""
        many_objects = ", ".join([f"object{i}" for i in range(15)])  # Exceeds 10 object limit
        is_valid, message = self.processor.validate_prompt(many_objects)
        assert not is_valid
        assert "too many" in message.lower()
    
    def test_validate_prompt_valid_punctuation(self):
        """Test that valid punctuation is allowed"""
        is_valid, message = self.processor.validate_prompt("person, car-truck, construction_worker")
        assert is_valid
    
    def test_case_insensitive_processing(self):
        """Test that processing is case insensitive"""
        result1 = self.processor.parse_detection_prompt("PERSON, CAR")
        result2 = self.processor.parse_detection_prompt("person, car")
        
        # Both should produce similar results (case normalized)
        assert result1[0].lower() == result2[0].lower()
        assert [obj.lower() for obj in result1[1]] == [obj.lower() for obj in result2[1]]
    
    def test_whitespace_handling(self):
        """Test proper whitespace handling"""
        grounding_prompt, objects = self.processor.parse_detection_prompt(
            "  person  ,  car with spaces  ,  dog  "
        )
        
        assert objects == ["person", "car with spaces", "dog"]
        assert "  " not in grounding_prompt  # No double spaces
    
    def test_complex_prompts(self):
        """Test complex real-world prompts"""
        prompts = [
            "construction worker with helmet and safety vest",
            "red car and blue truck in parking lot",
            "person wearing mask, laptop computer, coffee cup",
            "cat sitting on chair next to table"
        ]
        
        for prompt in prompts:
            is_valid, message = self.processor.validate_prompt(prompt)
            assert is_valid, f"Failed for prompt: {prompt}, message: {message}"
            
            grounding_prompt, objects = self.processor.parse_detection_prompt(prompt)
            assert len(objects) > 0
            assert grounding_prompt.endswith(".")
            assert " . " in grounding_prompt or grounding_prompt.count(".") == 1


class TestGlobalPromptProcessor:
    """Test the global prompt processor instance"""
    
    def test_global_instance_import(self):
        """Test that global instance can be imported"""
        from src.prompts import prompt_processor
        
        assert prompt_processor is not None
        assert hasattr(prompt_processor, 'validate_prompt')
        assert hasattr(prompt_processor, 'parse_detection_prompt')
    
    def test_global_instance_functionality(self):
        """Test that global instance works correctly"""
        from src.prompts import prompt_processor
        
        is_valid, message = prompt_processor.validate_prompt("person, car")
        assert is_valid
        
        grounding_prompt, objects = prompt_processor.parse_detection_prompt("person, car")
        assert objects == ["person", "car"]