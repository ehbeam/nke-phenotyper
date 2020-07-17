#!/usr/bin/python3

# Order of domains for plotting
order = {"data-driven": ["MEMORY", "REWARD", "COGNITION", 
						 "VISION", "MANIPULATION", "LANGUAGE"],
		 "rdoc": ["NEGATIVE_VALENCE", "POSITIVE_VALENCE", "AROUSAL_REGULATION", 
				  "SOCIAL_PROCESSES", "COGNITIVE_SYSTEMS", "SENSORIMOTOR_SYSTEMS"],
		 "dsm": ["DEPRESSIVE", "ANXIETY", "TRAUMA_STRESSOR", 
		 		 "OBSESSIVE_COMPULSIVE", "DISRUPTIVE", "SUBSTANCE",
		 		 "DEVELOPMENTAL", "PSYCHOTIC", "BIPOLAR"]}
order["dsm_diag"] = order["dsm"]

abbrev = {"data-driven": ["Memory", "Reward", "Cognition", 
						  "Vision", "Manipulation", "Language"],
		 "rdoc": ["Negative", "Positive", "Arousal & Reg.", 
				  "Social", "Cognitive", "Sensorimotor"],
		 "dsm": ["Depressive", "Anxiety", "Trauma & Stressor", 
		 		 "Obsess.-Compuls.", "Disruptive", "Substance",
		 		 "Developmental", "Psychotic", "Bipolar"]}
abbrev["dsm_diag"] = abbrev["dsm"]


# Font for plotting
# font = "style/Arial Unicode.ttf"
font = "style/cmu/cmunss.ttf"

# Hex color mappings
c = {"magenta": "#AA436A", "red": "#CA4F52", "vermillion": "#C16137", "brown": "#AC835B", "orange": "#E8B586", 
	 "gold": "#D19A17", "yellow": "#DCC447", "chartreuse": "#D9DC77", "lime": "#82B858", "green": "#43A971",
	 "teal": "#48A4A8", "blue": "#5B81BD", "indigo": "#7275B9", "purple": "#924DA0", "lavendar": "#D599DD"}

# Prespecified color order for each framework
fw2c = {"data-driven": ["blue", "vermillion", "yellow", "purple", "green", "gold"],
	    "rdoc": ["teal", "red", "chartreuse", "lavendar", "lime", "orange"],
	    "dsm": ["teal", "red", "chartreuse", "lavendar", "lime", "orange", "indigo", "magenta", "brown"]}
fw2c["dsm_diag"] = fw2c["dsm"]

# Palettes for general plotting
palettes = {fw: [c[color] for color in colors] for fw, colors in fw2c.items()}
palettes["dsm_diag"]
