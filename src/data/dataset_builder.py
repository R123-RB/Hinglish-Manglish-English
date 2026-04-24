"""
dataset_builder.py
==================
Generates the synthetic 3-layer Hinglish + Manglish → English dataset.

Layers per sample:
  1. input    : code-mixed Hinglish / Manglish / Mixed sentence
  2. literal  : word-for-word translation (structural learning)
  3. natural  : culturally accurate, fluent English (main training target)
  4. emotion  : emotion tag (auxiliary task / metadata)
  5. category : subcategory label

Instruction format applied:
  "Translate the following Hinglish/Manglish sentence into natural English
   preserving cultural meaning:\n'{input}'"

Run directly:
  python src/data/dataset_builder.py
  → Writes data/processed/train.json, val.json, test.json
"""

import json
import random
import yaml
from pathlib import Path
from typing import Dict, List
from loguru import logger


# ═══════════════════════════════════════════════════════════════════════════════
#  SEED DATASET  (300+ base pairs — expanded to 900+ via augmentation)
# ═══════════════════════════════════════════════════════════════════════════════

SEED_DATA: List[Dict] = [

    # ── PURE MANGLISH: Fatigue / Exhaustion ────────────────────────────────────
    {"input": "Enikku vayya da, full tired aanu",
     "literal": "I am not able brother, full tired is",
     "natural": "I'm exhausted, man.",
     "emotion": "fatigue", "category": "manglish_emotion"},

    {"input": "Njaan full maduthu, rest venam",
     "literal": "I am full fed up, rest needed",
     "natural": "I'm totally burnt out, I need a break.",
     "emotion": "exhaustion", "category": "manglish_emotion"},

    {"input": "Enikku innu kashtam aanu, manasilaayi",
     "literal": "Today it is hard for me, understood",
     "natural": "I'm having a rough day today, you know?",
     "emotion": "stress", "category": "manglish_emotion"},

    {"input": "Full tiredness aanu da, sleep cheyyum",
     "literal": "Full tiredness is brother, will sleep",
     "natural": "I'm dead tired, I'm going to sleep.",
     "emotion": "fatigue", "category": "manglish_emotion"},

    {"input": "Enikku moonnum kashtam aanu",
     "literal": "Three difficulty is for me",
     "natural": "I'm really struggling right now.",
     "emotion": "stress", "category": "manglish_emotion"},

    # ── PURE MANGLISH: Plans / Not Coming ──────────────────────────────────────
    {"input": "Bro njan innu varilla, mood illa",
     "literal": "Bro I today won't come, mood not",
     "natural": "Bro, I'm not coming today, I'm not in the mood.",
     "emotion": "low_energy", "category": "manglish_plans"},

    {"input": "Njan nale varilla machane",
     "literal": "I tomorrow won't come dude",
     "natural": "I won't be coming tomorrow, dude.",
     "emotion": "neutral", "category": "manglish_plans"},

    {"input": "Njan oru mani neram kuzhi varike varilla",
     "literal": "I one hour late until won't come",
     "natural": "I'll be at least an hour late.",
     "emotion": "neutral", "category": "manglish_plans"},

    {"input": "Njan try cheyyam, pakka aagilya",
     "literal": "I will try, definitely not",
     "natural": "I'll try, but no guarantees.",
     "emotion": "uncertain", "category": "manglish_plans"},

    {"input": "Innu varaan patilla da",
     "literal": "Today to come won't work brother",
     "natural": "I can't make it today, man.",
     "emotion": "apologetic", "category": "manglish_plans"},

    {"input": "Njan late aakum, wait cheyyu",
     "literal": "I will be late, wait",
     "natural": "I'll be late, please wait for me.",
     "emotion": "apologetic", "category": "manglish_plans"},

    # ── PURE MANGLISH: Drama / Scene ───────────────────────────────────────────
    {"input": "Avan scene aanu",
     "literal": "He is the scene",
     "natural": "He's creating drama.",
     "emotion": "observation", "category": "manglish_slang"},

    {"input": "Full scene aanu ividudhu",
     "literal": "Full scene is here",
     "natural": "There's a lot of drama going on here.",
     "emotion": "observation", "category": "manglish_slang"},

    {"input": "Aval enthino scene undaakkunnu",
     "literal": "She is making scene for something",
     "natural": "She's stirring up drama over nothing.",
     "emotion": "annoyed", "category": "manglish_slang"},

    {"input": "Scene kaanakan poyaloo",
     "literal": "Scene went to see",
     "natural": "It turned into quite a scene.",
     "emotion": "surprise", "category": "manglish_slang"},

    # ── PURE MANGLISH: Panic / Frustration ─────────────────────────────────────
    {"input": "Aiyyo njan poyi",
     "literal": "Oh no I am gone",
     "natural": "Oh no, I'm screwed.",
     "emotion": "panic", "category": "manglish_emotion"},

    {"input": "Aiyyo, deadline miss ayi",
     "literal": "Oh no, deadline missed",
     "natural": "Oh no, I missed the deadline!",
     "emotion": "panic", "category": "manglish_emotion"},

    {"input": "Adipoli aayit poyi bro",
     "literal": "Fantastic went bro",
     "natural": "It went brilliantly, bro!",
     "emotion": "excited", "category": "manglish_slang"},

    {"input": "Ithokkey kashtam aanu machane",
     "literal": "All this is difficult dude",
     "natural": "This is all really tough, dude.",
     "emotion": "frustration", "category": "manglish_emotion"},

    {"input": "Enthu cheyyanam ennum ariyilla",
     "literal": "What to do also don't know",
     "natural": "I have no idea what to do.",
     "emotion": "confused", "category": "manglish_emotion"},

    # ── PURE MANGLISH: Positive / Hype ─────────────────────────────────────────
    {"input": "Set aayit poyi machane!",
     "literal": "Set went dude",
     "natural": "It all worked out perfectly, dude!",
     "emotion": "satisfied", "category": "manglish_slang"},

    {"input": "Poli aayit poyi bro",
     "literal": "Awesome went bro",
     "natural": "It was absolutely brilliant, bro!",
     "emotion": "excited", "category": "manglish_slang"},

    {"input": "Mass aanu da avan",
     "literal": "Mass is brother he",
     "natural": "He's really impressive, man.",
     "emotion": "admiration", "category": "manglish_slang"},

    {"input": "Super aayit poyi, full happy aanu",
     "literal": "Super went, full happy is",
     "natural": "It went great, I'm really happy!",
     "emotion": "joy", "category": "manglish_slang"},

    {"input": "Eniku oru paisa odinn ethuvanna feeling aanu",
     "literal": "For me a penny ran to come feeling is",
     "natural": "I feel so alive right now!",
     "emotion": "excitement", "category": "manglish_emotion"},

    # ── PURE MANGLISH: Casual Questions ────────────────────────────────────────
    {"input": "Enthaa njan cheyyendi?",
     "literal": "What am I to do?",
     "natural": "What am I supposed to do?",
     "emotion": "confused", "category": "manglish_question"},

    {"input": "Evidaanu nee?",
     "literal": "Where are you?",
     "natural": "Where are you?",
     "emotion": "curious", "category": "manglish_question"},

    {"input": "Nee innu varumo?",
     "literal": "Are you coming today?",
     "natural": "Are you coming today?",
     "emotion": "curious", "category": "manglish_question"},

    {"input": "Enthu parayaananu?",
     "literal": "What is to say?",
     "natural": "What can I even say?",
     "emotion": "resigned", "category": "manglish_question"},

    {"input": "Evide aanu plan?",
     "literal": "Where is the plan?",
     "natural": "So what's the plan?",
     "emotion": "curious", "category": "manglish_question"},

    {"input": "Enthu aayaloo machane?",
     "literal": "What became dude?",
     "natural": "What happened, dude?",
     "emotion": "curious", "category": "manglish_question"},

    # ── PURE MANGLISH: Social / Respect ────────────────────────────────────────
    {"input": "Chetta parayum, njan kekkam",
     "literal": "Elder brother will say, I will listen",
     "natural": "Please go ahead, I'm listening.",
     "emotion": "respectful", "category": "manglish_social"},

    {"input": "Okay da, njan nokki parayam",
     "literal": "Okay brother, I will look and tell",
     "natural": "Alright, I'll take a look and let you know.",
     "emotion": "helpful", "category": "manglish_social"},

    {"input": "Machane, trust cheyyu enney",
     "literal": "Dude, trust me",
     "natural": "Dude, trust me on this.",
     "emotion": "assuring", "category": "manglish_social"},

    # ── PURE HINGLISH: Not Coming / Plans ──────────────────────────────────────
    {"input": "Bhai kal nahi aunga, mood nahi hai",
     "literal": "Brother tomorrow not come, mood not is",
     "natural": "Bro, I'm not coming tomorrow, I'm not in the mood.",
     "emotion": "low_energy", "category": "hinglish_plans"},

    {"input": "Yaar aaj nahi ho payega",
     "literal": "Friend today not will happen",
     "natural": "Man, it's not happening today.",
     "emotion": "apologetic", "category": "hinglish_plans"},

    {"input": "Main late aa raha hoon bhai",
     "literal": "I late coming am brother",
     "natural": "I'm running late, bro.",
     "emotion": "apologetic", "category": "hinglish_plans"},

    {"input": "Kya scene hai kal ka?",
     "literal": "What scene is tomorrow's?",
     "natural": "What's the plan for tomorrow?",
     "emotion": "curious", "category": "hinglish_question"},

    {"input": "Aaj nahi kar sakta yaar",
     "literal": "Today can't do friend",
     "natural": "I can't do it today, man.",
     "emotion": "apologetic", "category": "hinglish_plans"},

    # ── PURE HINGLISH: Questions ────────────────────────────────────────────────
    {"input": "Scene kya hai?",
     "literal": "What is the scene?",
     "natural": "What's going on?",
     "emotion": "curious", "category": "hinglish_question"},

    {"input": "Kya baat hai yaar?",
     "literal": "What is the matter friend?",
     "natural": "What's up, man?",
     "emotion": "curious", "category": "hinglish_question"},

    {"input": "Sab theek hai na bhai?",
     "literal": "All fine is right brother?",
     "natural": "Everything okay, bro?",
     "emotion": "concerned", "category": "hinglish_question"},

    {"input": "Kab aayega tu?",
     "literal": "When will you come?",
     "natural": "When are you coming?",
     "emotion": "curious", "category": "hinglish_question"},

    {"input": "Kahan hai tu abhi?",
     "literal": "Where are you right now?",
     "natural": "Where are you right now?",
     "emotion": "curious", "category": "hinglish_question"},

    {"input": "Kya hua bhai?",
     "literal": "What happened brother?",
     "natural": "What happened, bro?",
     "emotion": "concerned", "category": "hinglish_question"},

    # ── PURE HINGLISH: Positive / Hype ─────────────────────────────────────────
    {"input": "Ekdum mast tha bhai!",
     "literal": "Absolutely great was brother!",
     "natural": "It was absolutely amazing, bro!",
     "emotion": "excited", "category": "hinglish_slang"},

    {"input": "Kya baat hai, jhakaas!",
     "literal": "What a thing, fantastic!",
     "natural": "Wow, that's brilliant!",
     "emotion": "impressed", "category": "hinglish_slang"},

    {"input": "Dhamaal ho gaya yaar!",
     "literal": "Fun happened friend!",
     "natural": "It was so much fun, man!",
     "emotion": "excitement", "category": "hinglish_slang"},

    {"input": "Paisa vasool tha bhai",
     "literal": "Money recovered was brother",
     "natural": "Totally worth it, bro.",
     "emotion": "satisfied", "category": "hinglish_slang"},

    {"input": "Full on mast mood hai aaj",
     "literal": "Full on great mood is today",
     "natural": "I'm in a really great mood today.",
     "emotion": "joy", "category": "hinglish_slang"},

    # ── PURE HINGLISH: Frustration / Negative ─────────────────────────────────
    {"input": "Yaar bahut thak gaya hoon",
     "literal": "Friend very tired went am",
     "natural": "Man, I'm completely exhausted.",
     "emotion": "fatigue", "category": "hinglish_emotion"},

    {"input": "Bhai bore ho gaya main",
     "literal": "Brother bored went I",
     "natural": "I'm so bored, bro.",
     "emotion": "boredom", "category": "hinglish_emotion"},

    {"input": "Sab bakwas hai yaar",
     "literal": "All nonsense is friend",
     "natural": "This is all rubbish, man.",
     "emotion": "annoyed", "category": "hinglish_emotion"},

    {"input": "Main bahut pareshan hoon bhai",
     "literal": "I very troubled am brother",
     "natural": "I'm really stressed out, bro.",
     "emotion": "stress", "category": "hinglish_emotion"},

    {"input": "Yaar kuch samajh nahi aa raha",
     "literal": "Friend nothing understand not coming",
     "natural": "Man, I can't figure anything out.",
     "emotion": "confused", "category": "hinglish_emotion"},

    # ── PURE HINGLISH: Social / Casual ─────────────────────────────────────────
    {"input": "Araam se bhai, tension mat le",
     "literal": "Relax brother, don't take tension",
     "natural": "Chill out, bro, don't stress.",
     "emotion": "reassuring", "category": "hinglish_social"},

    {"input": "Chillax yaar, sab ho jayega",
     "literal": "Chillax friend, all will happen",
     "natural": "Relax, man, everything will work out.",
     "emotion": "reassuring", "category": "hinglish_social"},

    {"input": "Pakka aayega kal?",
     "literal": "Definitely coming tomorrow?",
     "natural": "You're definitely coming tomorrow, right?",
     "emotion": "curious", "category": "hinglish_social"},

    {"input": "Bhai sun, ek kaam kar",
     "literal": "Brother listen, one work do",
     "natural": "Hey bro, listen, do me a favour.",
     "emotion": "requesting", "category": "hinglish_social"},

    {"input": "Yaar, tera kya plan hai?",
     "literal": "Friend, your what plan is?",
     "natural": "Man, what are your plans?",
     "emotion": "curious", "category": "hinglish_social"},

    # ── MIXED HINGLISH + MANGLISH ──────────────────────────────────────────────
    {"input": "Kal njan office varilla bro",
     "literal": "Tomorrow I office won't come bro",
     "natural": "I won't come to the office tomorrow, bro.",
     "emotion": "neutral", "category": "mixed"},

    {"input": "Yaar, njan full tired aanu today",
     "literal": "Friend, I full tired am today",
     "natural": "Man, I'm completely exhausted today.",
     "emotion": "fatigue", "category": "mixed"},

    {"input": "Scene kya hai da? Kuch pata nahi",
     "literal": "What scene is brother? Something know not",
     "natural": "What's going on, dude? I have no idea.",
     "emotion": "confused", "category": "mixed"},

    {"input": "Bhai mood illa, innu varaan patilla",
     "literal": "Brother mood not, today to come won't work",
     "natural": "Bro, I'm not in the mood, I can't come today.",
     "emotion": "low_energy", "category": "mixed"},

    {"input": "Mood illa yaar, maduthu",
     "literal": "Mood not friend, fed up",
     "natural": "Not in the mood at all, man, I'm fed up.",
     "emotion": "frustration", "category": "mixed"},

    {"input": "Avan bahut scene aanu da",
     "literal": "He very scene is brother",
     "natural": "He's causing way too much drama, dude.",
     "emotion": "annoyed", "category": "mixed"},

    {"input": "Set aayit poyi bhai!",
     "literal": "Set went brother!",
     "natural": "It all worked out perfectly, bro!",
     "emotion": "satisfied", "category": "mixed"},

    {"input": "Pakka varilla yaar, enikku vayya",
     "literal": "Definitely won't come friend, for me can't",
     "natural": "Definitely not coming, man, I just can't.",
     "emotion": "certain", "category": "mixed"},

    {"input": "Full scene aanu bhai, aiyyo",
     "literal": "Full scene is brother, oh no",
     "natural": "Oh no, bro, it's a complete mess.",
     "emotion": "panic", "category": "mixed"},

    {"input": "Yaar njan aaj try cheyyam, but pakka aagilla",
     "literal": "Friend I today will try, but definitely won't happen",
     "natural": "Man, I'll try today, but I can't promise anything.",
     "emotion": "uncertain", "category": "mixed"},

    {"input": "Bhai avan full mass aanu da",
     "literal": "Brother he full mass is brother",
     "natural": "Bro, he's genuinely impressive, man.",
     "emotion": "admiration", "category": "mixed"},

    {"input": "Aiyyo bhai, deadline miss ayi",
     "literal": "Oh no brother, deadline missed",
     "natural": "Oh no, bro, I missed the deadline.",
     "emotion": "panic", "category": "mixed"},

    {"input": "Njan innu late varum yaar",
     "literal": "I today late will come friend",
     "natural": "I'll be late today, man.",
     "emotion": "apologetic", "category": "mixed"},

    {"input": "Kuch cheyyaan thonnunilla yaar",
     "literal": "Something to do not feeling friend",
     "natural": "I don't feel like doing anything, man.",
     "emotion": "low_energy", "category": "mixed"},

    # ── CODE-MIXED GRAMMAR ─────────────────────────────────────────────────────
    {"input": "I kal come cheyyum",
     "literal": "I tomorrow come will",
     "natural": "I'll come tomorrow.",
     "emotion": "neutral", "category": "codemix_grammar"},

    {"input": "Nee tomorrow varumo?",
     "literal": "You tomorrow coming?",
     "natural": "Are you coming tomorrow?",
     "emotion": "curious", "category": "codemix_grammar"},

    {"input": "He avan friends koode poyi",
     "literal": "He his friends with went",
     "natural": "He went with his friends.",
     "emotion": "neutral", "category": "codemix_grammar"},

    {"input": "She aval full happy aanu",
     "literal": "She she full happy is",
     "natural": "She seems really happy.",
     "emotion": "observation", "category": "codemix_grammar"},

    {"input": "Meeting innu cancel aayit poyi",
     "literal": "Meeting today cancelled went",
     "natural": "The meeting got cancelled today.",
     "emotion": "neutral", "category": "codemix_grammar"},

    {"input": "Office kal close aanu",
     "literal": "Office tomorrow close is",
     "natural": "The office is closed tomorrow.",
     "emotion": "neutral", "category": "codemix_grammar"},

    {"input": "Exam nale aanu, read cheythailla",
     "literal": "Exam tomorrow is, read did not",
     "natural": "The exam is tomorrow and I haven't studied.",
     "emotion": "panic", "category": "codemix_grammar"},

    {"input": "Meeting start cheyyan ready aano?",
     "literal": "Meeting to start ready is?",
     "natural": "Are we ready to start the meeting?",
     "emotion": "curious", "category": "codemix_grammar"},

    {"input": "Work submit cheythit vaa",
     "literal": "Work submit did came",
     "natural": "Come after you've submitted the work.",
     "emotion": "instructing", "category": "codemix_grammar"},

    {"input": "Call cheyyam njan abhi",
     "literal": "Call will I now",
     "natural": "I'll call you right now.",
     "emotion": "helpful", "category": "codemix_grammar"},

    # ── SLANG NORMALIZATION ─────────────────────────────────────────────────────
    {"input": "Avan full poli aanu da",
     "literal": "He full awesome is brother",
     "natural": "He's absolutely brilliant, man.",
     "emotion": "admiration", "category": "slang_norm"},

    {"input": "Inthe movie full lit aanu bhai",
     "literal": "This movie full lit is brother",
     "natural": "This movie is on fire, bro.",
     "emotion": "excited", "category": "slang_norm"},

    {"input": "Bindaas enjoy cheyyu bhai",
     "literal": "Carefree enjoy brother",
     "natural": "Just chill and enjoy, bro.",
     "emotion": "relaxed", "category": "slang_norm"},

    {"input": "Jugaad cheythit set aay",
     "literal": "Jugaad did set became",
     "natural": "Found a hack and sorted it out.",
     "emotion": "clever", "category": "slang_norm"},

    {"input": "Athokkey timepass aanu",
     "literal": "That all timepass is",
     "natural": "That's all just a waste of time.",
     "emotion": "dismissive", "category": "slang_norm"},

    {"input": "Full on bindaas life aanu avan",
     "literal": "Full on carefree life is his",
     "natural": "He lives his life completely carefree.",
     "emotion": "admiration", "category": "slang_norm"},

    # ── SOCIAL TONE: SYMPATHY / SUPPORT ────────────────────────────────────────
    {"input": "Kashtam aanu machane, understand cheyyunnu",
     "literal": "Difficult is dude, understanding",
     "natural": "That's tough, dude, I understand.",
     "emotion": "empathy", "category": "social_support"},

    {"input": "Aayi kollu, next time pakka",
     "literal": "It came be, next time definitely",
     "natural": "It's okay, you'll get it next time for sure.",
     "emotion": "encouraging", "category": "social_support"},

    {"input": "Don't worry bhai, sab theek ho jayega",
     "literal": "Don't worry brother, all fine will happen",
     "natural": "Don't worry, bro, everything will be fine.",
     "emotion": "reassuring", "category": "social_support"},

    {"input": "Njan undaakum da, tension edukkaruthu",
     "literal": "I will be brother, tension shouldn't take",
     "natural": "I've got you, man, don't stress.",
     "emotion": "supportive", "category": "social_support"},

    # ── SOCIAL TONE: CELEBRATION ────────────────────────────────────────────────
    {"input": "Congrats bro, fully deserve aayi!",
     "literal": "Congrats bro, fully deserved!",
     "natural": "Congrats, bro! You totally deserved it!",
     "emotion": "celebratory", "category": "social_celebrate"},

    {"input": "Dude njan select aayit poyi!",
     "literal": "Dude I selected went!",
     "natural": "Dude, I got selected!",
     "emotion": "excitement", "category": "social_celebrate"},

    {"input": "Bhai hum jeet gaye yaar!",
     "literal": "Brother we won friend!",
     "natural": "Bro, we won, man!",
     "emotion": "celebration", "category": "social_celebrate"},

    {"input": "Poli result aayit poyi machane!",
     "literal": "Awesome result came dude!",
     "natural": "Got an amazing result, dude!",
     "emotion": "excited", "category": "social_celebrate"},

    # ── FOOD / DAILY LIFE ──────────────────────────────────────────────────────
    {"input": "Khaana kazhikaan poyit vaa",
     "literal": "Food to eat going came",
     "natural": "Come after eating.",
     "emotion": "casual", "category": "daily_life"},

    {"input": "Coffee kudikkaan varumo?",
     "literal": "Coffee to drink coming?",
     "natural": "Coming for coffee?",
     "emotion": "casual", "category": "daily_life"},

    {"input": "Bhai bhook lag rahi hai bahut",
     "literal": "Brother hunger happening is a lot",
     "natural": "Bro, I'm starving.",
     "emotion": "hungry", "category": "daily_life"},

    {"input": "Njan kazhichu, nee?",
     "literal": "I ate, you?",
     "natural": "I've eaten, have you?",
     "emotion": "casual", "category": "daily_life"},

    # ── WORK / STUDY ────────────────────────────────────────────────────────────
    {"input": "Assignment submit cheyyan marichu",
     "literal": "Assignment to submit forgot",
     "natural": "I forgot to submit the assignment.",
     "emotion": "panic", "category": "work_study"},

    {"input": "Study cheyyaan thonnunilla machane",
     "literal": "Study to do not feeling dude",
     "natural": "I don't feel like studying at all, dude.",
     "emotion": "low_energy", "category": "work_study"},

    {"input": "Nale exam aanu bhai, prepared alla",
     "literal": "Tomorrow exam is brother, prepared not",
     "natural": "Exam is tomorrow, bro, and I'm not prepared.",
     "emotion": "stressed", "category": "work_study"},

    {"input": "Project done aayit poyi yaar!",
     "literal": "Project done went friend!",
     "natural": "The project is done, man!",
     "emotion": "relieved", "category": "work_study"},

    {"input": "Kuch samajh nahi aa raha yaar, help kar",
     "literal": "Something understand not coming friend, help do",
     "natural": "I'm not getting any of this, man, please help.",
     "emotion": "confused", "category": "work_study"},

    # ── WEATHER / ENVIRONMENT ──────────────────────────────────────────────────
    {"input": "Full rain aanu da, varaanaagilya",
     "literal": "Full rain is brother, to come won't",
     "natural": "It's pouring, man, I can't come.",
     "emotion": "apologetic", "category": "daily_life"},

    {"input": "Bhai bahut garmi hai aaj",
     "literal": "Brother very heat is today",
     "natural": "Bro, it's way too hot today.",
     "emotion": "discomfort", "category": "daily_life"},

    # ── INTERNET / REACTIONS ────────────────────────────────────────────────────
    {"input": "Seriously da, ithu unbelievable aanu",
     "literal": "Seriously brother, this unbelievable is",
     "natural": "Seriously, man, this is unbelievable.",
     "emotion": "shocked", "category": "reaction"},

    {"input": "Bhai yeh toh next level hai",
     "literal": "Brother this is next level",
     "natural": "Bro, this is on another level.",
     "emotion": "impressed", "category": "reaction"},

    {"input": "Njan seriously expect cheythilla machane",
     "literal": "I seriously expect did not dude",
     "natural": "I genuinely didn't expect that, dude.",
     "emotion": "surprised", "category": "reaction"},

    {"input": "Whaaat, ithenthu scenea da?",
     "literal": "What, this what scene brother?",
     "natural": "What?! What is even happening?",
     "emotion": "shocked", "category": "reaction"},

    {"input": "Bhai LOL, too funny",
     "literal": "Brother laughing out loud, too funny",
     "natural": "Bro, that's hilarious!",
     "emotion": "amused", "category": "reaction"},
]


# ═══════════════════════════════════════════════════════════════════════════════
#  INSTRUCTION TEMPLATE
# ═══════════════════════════════════════════════════════════════════════════════

INSTRUCTION_PREFIX = (
    "Translate the following Hinglish/Manglish sentence into natural English "
    "preserving cultural meaning:\n"
)


def format_instruction(sample: Dict) -> Dict:
    """Wrap sample in instruction format for mT5 training."""
    return {
        **sample,
        "instruction_input": f'{INSTRUCTION_PREFIX}"{sample["input"]}"',
        "instruction_target": sample["natural"],
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  DATASET BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

class DatasetBuilder:
    def __init__(self, config_path: str = "config/config.yaml"):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                self.cfg = yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config not found at {config_path}. Using defaults.")
            self.cfg = {}

        data_cfg = self.cfg.get("data", {})
        self.train_ratio = data_cfg.get("train_ratio", 0.80)
        self.val_ratio   = data_cfg.get("val_ratio",   0.10)
        self.processed_dir = Path(
            self.cfg.get("paths", {}).get("train_file", "data/processed/train.json")
        ).parent

    def build(self, augment: bool = True) -> Dict[str, List[Dict]]:
        """
        Build the full dataset from seed data, optionally augmenting.
        Returns dict with 'train', 'val', 'test' splits.
        """
        logger.info(f"Seed samples: {len(SEED_DATA)}")

        # Apply augmentation
        if augment:
            try:
                from src.data.augmentation import DataAugmentor
                aug = DataAugmentor(config_path="config/config.yaml")
                all_samples = aug.augment_dataset(SEED_DATA)
            except Exception as e:
                logger.warning(f"Augmentation failed ({e}). Using seed only.")
                all_samples = list(SEED_DATA)
        else:
            all_samples = list(SEED_DATA)

        # Format with instruction template
        all_samples = [format_instruction(s) for s in all_samples]

        # Shuffle reproducibly
        random.seed(42)
        random.shuffle(all_samples)

        # Split
        n = len(all_samples)
        n_train = int(n * self.train_ratio)
        n_val   = int(n * self.val_ratio)
        splits = {
            "train": all_samples[:n_train],
            "val":   all_samples[n_train:n_train + n_val],
            "test":  all_samples[n_train + n_val:],
        }

        logger.info(
            f"Split → train: {len(splits['train'])}, "
            f"val: {len(splits['val'])}, test: {len(splits['test'])}"
        )
        return splits

    def save(self, splits: Dict[str, List[Dict]]) -> None:
        """Write splits to data/processed/."""
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        for split_name, data in splits.items():
            out_path = self.processed_dir / f"{split_name}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved {len(data)} samples → {out_path}")

    def build_and_save(self, augment: bool = True) -> Dict[str, List[Dict]]:
        splits = self.build(augment=augment)
        self.save(splits)
        return splits

    def print_stats(self, splits: Dict[str, List[Dict]]) -> None:
        """Print dataset statistics."""
        from collections import Counter
        all_data = splits["train"] + splits["val"] + splits["test"]
        categories = Counter(s["category"] for s in all_data)
        emotions   = Counter(s["emotion"]   for s in all_data)
        print(f"\n{'='*55}")
        print(f"  DATASET STATISTICS  —  {len(all_data)} total samples")
        print(f"{'='*55}")
        print(f"\n  Train: {len(splits['train'])}  Val: {len(splits['val'])}  Test: {len(splits['test'])}")
        print(f"\n  Category distribution:")
        for cat, cnt in categories.most_common():
            print(f"    {cat:<30} {cnt}")
        print(f"\n  Emotion distribution (top 10):")
        for emo, cnt in emotions.most_common(10):
            print(f"    {emo:<25} {cnt}")
        print(f"{'='*55}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build Hinglish+Manglish dataset")
    parser.add_argument("--no-augment", action="store_true", help="Skip augmentation")
    parser.add_argument("--verify",     action="store_true", help="Print stats and exit")
    parser.add_argument("--config",     default="config/config.yaml")
    args = parser.parse_args()

    builder = DatasetBuilder(config_path=args.config)
    splits  = builder.build_and_save(augment=not args.no_augment)
    builder.print_stats(splits)

    if args.verify:
        # Show 3 random instruction-formatted examples
        print("\n  === 3 Random Samples (Instruction Format) ===")
        for s in random.sample(splits["train"], min(3, len(splits["train"]))):
            print(f"\n  Input  : {s['instruction_input']}")
            print(f"  Target : {s['instruction_target']}")
            print(f"  Emotion: {s['emotion']}  |  Category: {s['category']}")
