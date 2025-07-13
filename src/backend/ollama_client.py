# src/backend/ollama_client.py
import ollama
import base64
import io
from PIL import Image
import json

class EcoSentinelAI:
    def __init__(self, model_name="gemma3n:4b"):
        self.model_name = model_name
        self.client = ollama.Client()
        
    def analyze_plant_health(self, image_path, additional_info=""):
        """Analyze plant health from image"""
        image_b64 = self._encode_image(image_path)
        
        prompt = f"""
        Analyze this plant image for health assessment:
        
        1. Identify the plant species if possible
        2. Detect any diseases, pests, or nutrient deficiencies
        3. Assess overall plant health (scale 1-10)
        4. Provide treatment recommendations
        5. Suggest preventive measures
        
        Additional context: {additional_info}
        
        Format response as JSON with fields: species, health_score, issues, treatments, prevention
        """
        
        response = self.client.generate(
            model=self.model_name,
            prompt=prompt,
            images=[image_b64],
            stream=False
        )
        
        return self._parse_json_response(response['response'])
    
    def assess_environmental_hazards(self, image_path, location=""):
        """Assess environmental hazards from landscape image"""
        image_b64 = self._encode_image(image_path)
        
        prompt = f"""
        Analyze this environmental image for potential hazards:
        
        1. Wildfire risk assessment (scale 1-10)
        2. Flood risk indicators
        3. Air quality visual indicators
        4. Vegetation stress signs
        5. Immediate safety concerns
        
        Location context: {location}
        
        Format as JSON: wildfire_risk, flood_risk, air_quality, vegetation_health, safety_alerts
        """
        
        response = self.client.generate(
            model=self.model_name,
            prompt=prompt,
            images=[image_b64],
            stream=False
        )
        
        return self._parse_json_response(response['response'])
    
    def emergency_communication(self, message, target_language="English"):
        """Translate emergency information"""
        prompt = f"""
        Translate this emergency message to {target_language} and provide:
        
        1. Accurate translation
        2. Cultural context adjustments
        3. Urgency level (1-5)
        4. Recommended actions
        
        Original message: {message}
        
        Format as JSON: translation, urgency, actions, cultural_notes
        """
        
        response = self.client.generate(
            model=self.model_name,
            prompt=prompt,
            stream=False
        )
        
        return self._parse_json_response(response['response'])
    
    def identify_species(self, image_path, ecosystem_type=""):
        """Identify species for biodiversity monitoring"""
        image_b64 = self._encode_image(image_path)
        
        prompt = f"""
        Identify the species in this image:
        
        1. Species name (scientific and common)
        2. Conservation status
        3. Ecosystem role
        4. Population trends
        5. Conservation recommendations
        
        Ecosystem context: {ecosystem_type}
        
        Format as JSON: species_name, scientific_name, conservation_status, ecosystem_role, recommendations
        """
        
        response = self.client.generate(
            model=self.model_name,
            prompt=prompt,
            images=[image_b64],
            stream=False
        )
        
        return self._parse_json_response(response['response'])
    
    def _encode_image(self, image_path):
        """Convert image to base64 for Ollama"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def _parse_json_response(self, response_text):
        """Parse JSON from AI response"""
        try:
            # Extract JSON from response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            if start_idx != -1 and end_idx != -1:
                json_str = response_text[start_idx:end_idx]
                return json.loads(json_str)
            else:
                return {"raw_response": response_text}
        except:
            return {"raw_response": response_text}