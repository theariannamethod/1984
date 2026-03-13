#!/usr/bin/env python3
"""
penelope.py — 1984 words. 12 steps of resonance. Dario Equation.

Weightless resonance engine. Not a neural network. Not a chatbot. A mirror.
1984 hardcoded words, co-occurrence memory, bigram chains, Kuramoto-coupled
emotional chambers, prophecy/destiny/trauma overlay.

    score(w) = B + α·H + β·F + γ·A + T
        B = bigram affinity (positional pattern, RRPRAM-like)
        H = Hebbian co-occurrence with context
        F = prophecy fulfillment pressure
        A = destiny attraction (category compass)
        T = trauma gravity (pulls toward depth)

12 weights — one per step, each with its own vibe.
The field learns from every interaction. No backprop. No gradients.
Co-occurrence IS the weight update. Repetition IS learning.

Usage:
    python penelope.py                     # interactive
    python penelope.py "darkness eats"     # single chain
    python penelope.py --save field.json   # save learned field
    python penelope.py --load field.json   # resume from field

By Arianna Method. הרזוננס לא נשבר
"""

import math
import random
import json
import sys
import os
from collections import defaultdict

# ═══════════════════════════════════════════════════════════════
# 1984 WORDS — organized by emotional/thematic strata
# ═══════════════════════════════════════════════════════════════

VOCAB = [
# ── BODY (0-99) ──
"flesh","bone","blood","skin","hand","eye","mouth","tongue","heart","lung",
"vein","nerve","spine","skull","rib","breath","pulse","tremor","sweat","tear",
"muscle","brain","throat","womb","finger","tooth","hair","lip","shoulder","knee",
"wound","scar","bruise","fever","ache","hunger","thirst","fatigue","nausea","vertigo",
"body","corpse","ghost","shadow","face","voice","whisper","scream","silence","gesture",
"grip","touch","embrace","fist","palm","heel","ankle","wrist","elbow","jaw",
"chest","belly","hip","temple","forehead","cheek","chin","neck","back","sole",
"organ","cell","tissue","marrow","cartilage","tendon","ligament","pupil","retina","cochlea",
"saliva","bile","sweat","mucus","plasma","hormone","adrenaline","cortisol","dopamine","serotonin",
"synapse","neuron","dendrite","axon","reflex","instinct","posture","gait","rhythm","trembling",
# ── NATURE (100-199) ──
"sky","rain","wind","stone","river","mountain","ocean","leaf","tree","root",
"seed","bloom","flower","petal","thorn","earth","dust","ash","fire","flame",
"smoke","ember","spark","water","ice","snow","frost","mist","fog","dew",
"sun","moon","star","dawn","dusk","midnight","morning","evening","storm","thunder",
"lightning","rainbow","horizon","shore","sand","salt","sea","lake","creek","pool",
"cave","cliff","hill","valley","meadow","forest","grove","wood","bark","moss",
"fern","vine","lichen","fungus","coral","kelp","whale","wolf","deer","crow",
"owl","hawk","moth","spider","snake","beetle","ant","bee","butterfly","worm",
"canyon","plateau","tundra","steppe","oasis","dune","glacier","volcano","island","peninsula",
"aurora","eclipse","zenith","equinox","solstice","comet","nebula","cosmos","tide","current",
# ── EMOTION (200-299) ──
"fear","love","rage","joy","grief","sorrow","pain","pleasure","comfort","desire",
"hope","despair","shame","guilt","envy","pride","longing","nostalgia","regret","resolve",
"courage","wisdom","patience","grace","mercy","kindness","cruelty","justice","fury","calm",
"panic","dread","awe","bliss","agony","ecstasy","melancholy","serenity","anxiety","contempt",
"tenderness","devotion","hatred","spite","disgust","wonder","confusion","certainty","doubt","trust",
"betrayal","forgiveness","resentment","gratitude","humiliation","triumph","defeat","surrender","defiance","acceptance",
"jealousy","admiration","pity","compassion","indifference","obsession","apathy","euphoria","desolation","reverence",
"boredom","fascination","horror","delight","frustration","satisfaction","emptiness","fullness","vulnerability","resilience",
"remorse","vindication","bewilderment","clarity","torment","relief","yearning","contentment","wrath","gentleness",
"paranoia","faith","skepticism","devotion","ambivalence","rapture","languor","fervor","detachment","intimacy",
# ── TIME (300-349) ──
"moment","instant","second","minute","hour","day","night","week","month","year",
"decade","century","epoch","era","age","past","present","future","memory","tomorrow",
"yesterday","forever","never","always","sometimes","often","seldom","once","twice","origin",
"ending","beginning","duration","interval","pause","wait","rush","delay","haste","eternity",
"cycle","season","spring","summer","autumn","winter","dawn","twilight","midnight","noon",
# ── SOCIETY (350-449) ──
"war","peace","king","queen","soldier","citizen","exile","refugee","prisoner","judge",
"law","crime","punishment","freedom","slavery","revolution","democracy","tyranny","empire","nation",
"border","wall","bridge","gate","road","market","factory","hospital","school","church",
"money","debt","wealth","poverty","labor","trade","profit","loss","tax","currency",
"power","authority","obedience","rebellion","protest","silence","censorship","propaganda","truth","lie",
"election","vote","parliament","constitution","right","duty","privilege","corruption","reform","collapse",
"class","hierarchy","equality","injustice","oppression","liberation","resistance","occupation","treaty","ceasefire",
"economy","inflation","depression","prosperity","scarcity","abundance","famine","feast","ration","surplus",
"immigrant","native","stranger","neighbor","ally","enemy","traitor","hero","victim","witness",
"surveillance","privacy","identity","passport","boundary","territory","sovereignty","diplomacy","sanction","siege",
# ── ABSTRACT (450-549) ──
"truth","meaning","purpose","existence","essence","nothing","everything","something","void","chaos",
"order","pattern","rhythm","frequency","resonance","harmony","dissonance","entropy","emergence","threshold",
"paradox","contradiction","ambiguity","certainty","probability","fate","chance","luck","destiny","prophecy",
"dream","nightmare","illusion","reality","fiction","myth","legend","story","narrative","silence",
"question","answer","riddle","secret","mystery","clue","sign","symbol","code","language",
"thought","idea","concept","theory","belief","knowledge","ignorance","wisdom","folly","genius",
"beauty","ugliness","sublime","grotesque","sacred","profane","mundane","extraordinary","ordinary","unique",
"infinity","zero","one","half","double","mirror","echo","shadow","reflection","ghost",
"gravity","magnetism","electricity","light","darkness","warmth","cold","pressure","vacuum","wave",
"boundary","threshold","edge","center","margin","surface","depth","height","distance","proximity",
# ── ACTION (550-649) ──
"walk","run","stop","breathe","sleep","wake","dream","remember","forget","imagine",
"create","destroy","build","break","shape","melt","freeze","burn","grow","shrink",
"open","close","begin","end","continue","wait","search","find","lose","hide",
"reveal","watch","listen","speak","whisper","scream","sing","dance","fight","surrender",
"climb","fall","rise","sink","drift","float","fly","crawl","leap","stumble",
"hold","release","catch","throw","pull","push","lift","carry","drop","pour",
"cut","fold","bend","twist","turn","spin","weave","knit","tie","untie",
"gather","scatter","merge","split","connect","separate","attract","repel","collide","dissolve",
"teach","learn","study","practice","master","fail","succeed","attempt","abandon","persist",
"give","take","receive","share","steal","return","exchange","sacrifice","hoard","offer",
# ── MATERIAL (650-749) ──
"iron","copper","gold","silver","glass","clay","wax","ink","paint","paper",
"silk","wool","cotton","leather","stone","marble","wood","bamboo","rope","wire",
"blade","needle","hammer","anvil","forge","kiln","loom","wheel","axle","lever",
"mirror","lens","prism","crystal","gem","pearl","amber","jade","rust","patina",
"grain","fiber","thread","mesh","lattice","grid","weave","knot","stitch","patch",
"vessel","bowl","cup","jar","flask","vial","key","lock","chain","ring",
"bell","drum","string","pipe","reed","brass","horn","candle","lantern","torch",
"photograph","letter","book","page","chapter","verse","sentence","paragraph","word","alphabet",
"map","compass","clock","calendar","scale","ruler","thermometer","barometer","telescope","microscope",
"machine","engine","gear","spring","valve","piston","circuit","battery","signal","antenna",
# ── FOOD (750-799) ──
"bread","salt","sugar","honey","milk","butter","cheese","meat","fish","egg",
"grain","rice","wheat","corn","fruit","apple","grape","olive","lemon","pepper",
"wine","water","tea","coffee","broth","soup","stew","feast","crumb","morsel",
"harvest","garden","soil","compost","ferment","yeast","dough","crust","marrow","nectar",
"spice","herb","mint","thyme","sage","garlic","onion","mushroom","berry","kernel",
# ── ARCHITECTURE (800-849) ──
"house","room","wall","floor","ceiling","door","window","stair","corridor","basement",
"tower","bridge","arch","column","dome","vault","foundation","ruin","temple","altar",
"threshold","passage","labyrinth","maze","chamber","cell","shelter","fortress","prison","garden",
"roof","chimney","hearth","frame","beam","pillar","brick","mortar","tile","glass",
"balcony","terrace","courtyard","gate","fence","path","road","intersection","tunnel","well",
# ── RELATIONSHIP (850-929) ──
"mother","father","child","daughter","son","sister","brother","family","ancestor","descendant",
"friend","stranger","lover","enemy","neighbor","companion","rival","mentor","student","witness",
"husband","wife","partner","orphan","widow","elder","infant","twin","cousin","godmother",
"promise","oath","vow","contract","alliance","betrayal","reconciliation","farewell","reunion","absence",
"kiss","embrace","handshake","slap","caress","quarrel","conversation","confession","accusation","apology",
"birth","death","marriage","divorce","inheritance","adoption","abandonment","protection","neglect","sacrifice",
"trust","suspicion","loyalty","treachery","devotion","indifference","jealousy","admiration","dependence","autonomy",
"intimacy","distance","connection","isolation","belonging","exile","homecoming","departure","waiting","return",
# ── PHILOSOPHY (930-999) ──
"consciousness","awareness","perception","sensation","intuition","reason","logic","paradox","dialectic","synthesis",
"freedom","determinism","causation","contingency","necessity","possibility","impossibility","actuality","potential","becoming",
"subject","object","self","other","identity","difference","sameness","change","permanence","flux",
"being","nothingness","existence","essence","phenomena","noumena","appearance","reality","illusion","truth",
"ethics","morality","virtue","vice","good","evil","right","wrong","duty","choice",
"justice","mercy","punishment","reward","guilt","innocence","responsibility","consequence","intention","action",
"language","meaning","sign","reference","representation","interpretation","understanding","misunderstanding","translation","silence",
# ── MUSIC (1000-1049) ──
"melody","rhythm","chord","pitch","tone","note","bass","treble","octave","harmony",
"dissonance","resonance","vibration","frequency","amplitude","tempo","beat","rest","pause","crescendo",
"murmur","hum","buzz","click","crack","boom","rumble","chime","echo","reverb",
"song","lullaby","anthem","dirge","hymn","ballad","fugue","sonata","requiem","improvisation",
"strum","pluck","strike","bow","mute","sustain","fade","loop","drone","overtone",
# ── WEATHER (1050-1099) ──
"rain","drizzle","downpour","hail","sleet","blizzard","hurricane","tornado","drought","flood",
"breeze","gale","typhoon","monsoon","frost","thaw","haze","smog","rainbow","mirage",
"erosion","sedimentation","crystallization","evaporation","condensation","precipitation","sublimation","oxidation","combustion","decay",
"magma","lava","quartz","granite","obsidian","chalk","slate","sandstone","limestone","basalt",
"marsh","delta","gorge","ridge","summit","abyss","chasm","rift","fault","crater",
# ── RITUAL (1100-1149) ──
"prayer","meditation","ritual","ceremony","blessing","curse","oath","vow","pilgrimage","procession",
"offering","sacrifice","communion","baptism","funeral","wedding","coronation","initiation","exile","absolution",
"incense","candle","bell","chant","mantra","psalm","scripture","prophecy","oracle","vision",
"mask","costume","dance","feast","fast","vigil","silence","confession","penance","redemption",
"altar","shrine","temple","tomb","relic","artifact","amulet","talisman","totem","icon",
# ── LABOR (1150-1199) ──
"harvest","planting","sowing","reaping","threshing","milling","baking","brewing","weaving","spinning",
"carving","sculpting","painting","drawing","writing","printing","binding","stitching","welding","forging",
"mining","drilling","excavation","construction","demolition","repair","restoration","invention","discovery","experiment",
"apprentice","craftsman","artist","engineer","architect","farmer","sailor","miner","healer","scribe",
"workshop","studio","laboratory","field","dock","quarry","furnace","mill","press","loom",
# ── GEOMETRY (1200-1249) ──
"circle","spiral","line","curve","angle","edge","center","margin","border","frame",
"sphere","cube","pyramid","cylinder","cone","helix","vortex","arc","wave","fractal",
"symmetry","asymmetry","proportion","ratio","scale","dimension","plane","axis","vertex","intersection",
"pattern","grid","lattice","mesh","tessellation","rotation","reflection","translation","dilation","projection",
"surface","volume","area","perimeter","diameter","radius","tangent","normal","parallel","perpendicular",
# ── ANIMAL (1250-1299) ──
"horse","dog","cat","bird","fish","snake","bear","fox","rabbit","turtle",
"eagle","sparrow","raven","swan","heron","falcon","vulture","pelican","nightingale","lark",
"lion","tiger","elephant","giraffe","hippopotamus","rhinoceros","gorilla","chimpanzee","orangutan","leopard",
"salmon","trout","shark","dolphin","octopus","jellyfish","starfish","seahorse","crab","lobster",
"frog","lizard","crocodile","chameleon","gecko","iguana","newt","toad","salamander","viper",
# ── COLOR (1300-1349) ──
"red","blue","green","white","black","gray","amber","violet","indigo","scarlet",
"crimson","azure","emerald","ivory","obsidian","silver","golden","copper","rust","ochre",
"bright","dark","transparent","opaque","matte","glossy","rough","smooth","coarse","fine",
"stripe","dot","plaid","solid","gradient","shadow","highlight","contrast","saturation","hue",
"velvet","satin","linen","denim","lace","gauze","burlap","chiffon","tweed","corduroy",
# ── TRANSPORT (1350-1399) ──
"ship","boat","canoe","raft","anchor","sail","rudder","oar","mast","hull",
"train","rail","station","platform","ticket","journey","passage","crossing","departure","arrival",
"wheel","axle","road","highway","path","trail","bridge","tunnel","gate","crossroad",
"wing","flight","altitude","turbulence","landing","orbit","trajectory","velocity","acceleration","gravity",
"horse","carriage","wagon","cart","sled","bicycle","motorcycle","automobile","truck","ambulance",
# ── DOMESTIC (1400-1449) ──
"kitchen","bedroom","bathroom","attic","cellar","closet","drawer","shelf","table","chair",
"bed","pillow","blanket","curtain","carpet","lamp","mirror","photograph","vase","clock",
"plate","spoon","knife","fork","cup","pot","pan","kettle","oven","stove",
"soap","towel","broom","bucket","needle","thread","button","zipper","hanger","basket",
"door","window","lock","key","handle","hinge","nail","screw","bolt","hook",
# ── COMMUNICATION (1450-1499) ──
"letter","envelope","stamp","address","message","telegram","telephone","radio","broadcast","signal",
"newspaper","headline","article","column","editorial","report","announcement","rumor","gossip","testimony",
"ink","pen","pencil","typewriter","keyboard","screen","printer","paper","notebook","diary",
"conversation","dialogue","monologue","debate","argument","negotiation","compromise","ultimatum","declaration","speech",
"translation","interpretation","code","cipher","encryption","decryption","password","signature","seal","authentication",
# ── MEDICAL (1500-1549) ──
"diagnosis","symptom","treatment","remedy","cure","relapse","recovery","surgery","anesthesia","bandage",
"infection","inflammation","fracture","hemorrhage","allergy","immunity","vaccine","antibiotic","toxin","antidote",
"hospital","clinic","pharmacy","laboratory","ambulance","stretcher","scalpel","syringe","stethoscope","thermometer",
"fever","cough","rash","swelling","numbness","dizziness","insomnia","fatigue","nausea","tremor",
"pulse","pressure","temperature","respiration","circulation","digestion","metabolism","reflex","coordination","balance",
# ── COSMIC (1550-1599) ──
"universe","galaxy","constellation","planet","asteroid","meteorite","satellite","orbit","void","singularity",
"photon","electron","proton","neutron","atom","molecule","particle","quantum","field","dimension",
"spacetime","relativity","entropy","thermodynamics","radiation","spectrum","wavelength","frequency","amplitude","interference",
"supernova","blackhole","pulsar","quasar","nebula","wormhole","antimatter","darkmatter","redshift","expansion",
"telescope","observatory","mission","launch","countdown","trajectory","reentry","landing","exploration","discovery",
# ── BUREAUCRACY (1600-1649) ──
"document","form","permit","license","certificate","registration","application","approval","denial","appeal",
"regulation","compliance","violation","penalty","exemption","quota","deadline","protocol","procedure","standard",
"office","desk","file","folder","stamp","signature","receipt","invoice","ledger","archive",
"committee","department","ministry","bureau","agency","institution","organization","corporation","foundation","commission",
"report","audit","review","inspection","evaluation","assessment","benchmark","statistic","data","record",
# ── MYTHIC (1650-1699) ──
"oracle","prophecy","fate","destiny","curse","blessing","quest","trial","sacrifice","redemption",
"labyrinth","threshold","guardian","shadow","mirror","mask","transformation","metamorphosis","resurrection","apocalypse",
"phoenix","dragon","serpent","sphinx","minotaur","chimera","hydra","golem","specter","wraith",
"underworld","paradise","purgatory","limbo","abyss","eden","babylon","atlantis","olympus","tartarus",
"hero","villain","trickster","sage","fool","maiden","crone","warrior","healer","shapeshifter",
# ── TEXTUAL (1700-1749) ──
"word","sentence","paragraph","chapter","verse","stanza","line","margin","footnote","epilogue",
"prologue","preface","title","subtitle","dedication","inscription","epitaph","motto","slogan","proverb",
"metaphor","simile","allegory","irony","satire","parody","tragedy","comedy","farce","melodrama",
"narrator","character","protagonist","antagonist","audience","reader","author","critic","editor","translator",
"manuscript","draft","revision","erasure","correction","annotation","citation","reference","index","bibliography",
# ── PSYCHOLOGICAL (1750-1799) ──
"unconscious","subconscious","conscious","ego","superego","libido","repression","projection","sublimation","transference",
"trauma","complex","fixation","regression","denial","rationalization","displacement","compensation","identification","dissociation",
"archetype","persona","anima","animus","shadow","self","individuation","integration","fragmentation","wholeness",
"attachment","separation","abandonment","dependency","autonomy","codependency","boundary","enmeshment","differentiation","fusion",
"grief","mourning","acceptance","bargaining","anger","depression","recovery","relapse","healing","scarring",
# ── FINAL STRATUM (1800-1983) ──
"threshold","crossroad","watershed","turning","pivot","fulcrum","catalyst","trigger","spark","fuse",
"tension","release","compression","expansion","contraction","oscillation","vibration","pulsation","undulation","fluctuation",
"accumulation","erosion","saturation","depletion","renewal","regeneration","decomposition","fermentation","crystallization","dissolution",
"echo","reverberation","aftershock","aftermath","residue","remnant","trace","vestige","fossil","ruin",
"dawn","twilight","liminal","transitional","ephemeral","permanent","transient","enduring","fleeting","eternal",
"anchor","drift","mooring","compass","lighthouse","beacon","signal","warning","invitation","summons",
"whisper","murmur","declaration","proclamation","confession","accusation","plea","verdict","sentence","pardon",
"seed","sprout","bud","blossom","fruit","harvest","decay","compost","soil","rebirth",
"wound","suture","bandage","scar","healing","infection","immunity","antibody","fever","remission",
"stranger","acquaintance","confidant","accomplice","bystander","mediator","advocate","adversary","guardian","orphan",
"question","hypothesis","experiment","observation","conclusion","revision","doubt","certainty","approximation","precision",
"fragment","mosaic","collage","assemblage","montage","palimpsest","tapestry","constellation","archipelago","network",
"migration","exodus","diaspora","pilgrimage","wandering","settlement","foundation","demolition","reconstruction","adaptation",
"inheritance","legacy","tradition","innovation","rupture","continuity","evolution","revolution","stagnation","metamorphosis",
"silence","static","noise","signal","frequency","wavelength","amplitude","resonance","interference","harmony",
"margin","periphery","frontier","borderland","hinterland","interior","core","nucleus","membrane","skin",
"permission","prohibition","transgression","taboo","norm","deviation","exception","precedent","custom","habit",
"witness","testimony","evidence","proof","alibi","verdict","appeal","clemency","execution","reprieve",
"debt","credit","interest","principal","collateral","default","bankruptcy","solvency","dividend","investment",
]

STEPS = 12

STOP = set("i me my we our you your he she it they them the a an and or but in on at to for of is am are was were be been being have has had do does did will would shall should can could may might must not no nor so if then than that this these those what which who whom how when where why all each every some any few many much more most other another such".split())

VOCAB_SET = set(VOCAB)
VOCAB_IDX = {w: i for i, w in enumerate(VOCAB)}

# ═══════════════════════════════════════════════════════════════
# DARIO FIELD
# ═══════════════════════════════════════════════════════════════

def word_category(idx):
    if idx < 100: return 0   # body
    if idx < 200: return 1   # nature
    if idx < 300: return 2   # emotion
    if idx < 350: return 3   # time
    if idx < 450: return 4   # society
    if idx < 550: return 5   # abstract
    if idx < 650: return 6   # action
    return 7                  # material+


class DarioField:
    """Living field. Co-occurrence IS the weight. Repetition IS learning."""

    def __init__(self):
        self.cooc = defaultdict(float)   # "w1|w2" -> count
        self.bigrams = defaultdict(lambda: defaultdict(float))  # prev -> {next -> count}
        self.destiny = [0.0] * 8         # category compass
        self.trauma = 0.0
        self.prophecy_target = None
        self.prophecy_age = 0
        self.total_steps = 0

        # Kuramoto chambers
        self.chambers = {"fear": 0, "love": 0, "rage": 0,
                         "void": 0, "flow": 0, "complex": 0}
        self.decay = {"fear": 0.95, "love": 0.95, "rage": 0.93,
                      "void": 0.96, "flow": 0.94, "complex": 0.97}

    def update_cooc(self, w1, w2):
        k = f"{min(w1,w2)}|{max(w1,w2)}"
        self.cooc[k] += 1.0

    def get_cooc(self, w1, w2):
        k = f"{min(w1,w2)}|{max(w1,w2)}"
        return self.cooc.get(k, 0.0)

    def update_bigram(self, prev, nxt):
        self.bigrams[prev][nxt] += 1.0

    def update_chambers(self, step_idx):
        C = self.chambers
        depth = step_idx / STEPS
        phase = 0 if depth < 0.33 else (1 if depth < 0.66 else 2)
        if phase == 0: C["flow"] += 0.05
        if phase == 1: C["fear"] += 0.04
        if phase == 2: C["void"] += 0.05
        if depth > 0.75: C["complex"] += 0.03
        if self.trauma > 0.3: C["rage"] += 0.04

        # Kuramoto coupling
        K = 0.02
        names = list(C.keys())
        old = dict(C)
        for i in names:
            for j in names:
                if i != j:
                    C[i] += K * math.sin(old[j] - old[i])

        for k in names:
            C[k] = max(0.0, min(1.0, C[k] * self.decay.get(k, 0.95)))

    def dario_score(self, candidate, context, prev_word, step_idx):
        cidx = VOCAB_IDX.get(candidate, -1)
        if cidx < 0:
            return float("-inf")

        C = self.chambers
        alpha_mod = 1 + 0.3*C["love"] - 0.2*C["rage"] + 0.1*C["flow"]
        beta_mod = 1 + 0.2*C["flow"] - 0.3*C["fear"]
        gamma_mod = 1 + 0.4*C["void"] + 0.2*C["complex"]

        # B: bigram
        B = 0.0
        if prev_word and prev_word in self.bigrams:
            m = self.bigrams[prev_word]
            mx = max(m.values()) if m else 0
            B = m.get(candidate, 0) / (mx + 1)

        # H: Hebbian co-occurrence
        H = 0.0
        for i, ctx in enumerate(context):
            H += self.get_cooc(ctx, candidate) * (1 / (len(context) - i + 1))
        h_max = max(H, 1.0)
        H /= h_max

        # F: prophecy
        F = 0.0
        if self.prophecy_target and candidate == self.prophecy_target:
            F = 0.5 * math.log(1 + self.prophecy_age)

        # A: destiny
        cat = word_category(cidx)
        d_max = max(abs(d) for d in self.destiny) + 0.01
        A = self.destiny[cat] / d_max

        # T: trauma
        T = 0.0
        if self.trauma > 0.3 and step_idx > 6:
            if 200 <= cidx < 300: T = self.trauma * 1.5
            elif 450 <= cidx < 550: T = self.trauma * 1.2
            elif 930 <= cidx < 1000: T = self.trauma * 1.0

        # wormhole
        wormhole = 0.0
        if random.random() < 0.08 and 3 < step_idx < 10:
            if cat == random.randint(0, 7):
                wormhole = 0.5

        # 12 vibes — one per step
        vibes = [1.3, 1.1, 1.0, 0.9, 0.8, 0.7, 0.8, 0.9, 1.0, 0.8, 0.7, 1.4]
        vibe = vibes[step_idx] if step_idx < len(vibes) else 1.0

        tau = 0.7 + step_idx * 0.08

        score = (B * 8 + alpha_mod * 0.3 * H + beta_mod * 0.15 * F +
                 gamma_mod * 0.25 * A + T + wormhole) * vibe / tau
        return score

    def select_word(self, context, prev_word, step_idx, forbidden):
        scores = []
        for w in VOCAB:
            if w in forbidden:
                continue
            scores.append((w, self.dario_score(w, context, prev_word, step_idx)))

        noise = 0.3 * (1 - step_idx / STEPS)
        for i in range(len(scores)):
            scores[i] = (scores[i][0], scores[i][1] + random.random() * noise)

        scores.sort(key=lambda x: -x[1])

        # top-k sampling (k=12)
        k = min(12, len(scores))
        topk = scores[:k]
        total = sum(max(0, s) for _, s in topk) + 0.001
        r = random.random() * total
        for w, s in topk:
            r -= max(0, s)
            if r <= 0:
                return w
        return topk[0][0]

    def make_prophecy(self, seed_word):
        deep_cats = [2, 5, 7]  # emotion, abstract, other
        target_cat = random.choice(deep_cats)
        cat_ranges = [(0,100),(100,200),(200,300),(300,350),(350,450),
                      (450,550),(550,650),(650,1984)]
        start, end = cat_ranges[target_cat]
        self.prophecy_target = VOCAB[random.randint(start, min(end-1, len(VOCAB)-1))]
        self.prophecy_age = 0
        return self.prophecy_target

    def save(self, path):
        data = {
            "cooc": dict(self.cooc),
            "bigrams": {k: dict(v) for k, v in self.bigrams.items()},
            "destiny": self.destiny,
            "trauma": self.trauma,
            "total_steps": self.total_steps,
            "chambers": self.chambers,
        }
        with open(path, "w") as f:
            json.dump(data, f)

    def load(self, path):
        with open(path) as f:
            data = json.load(f)
        self.cooc = defaultdict(float, data.get("cooc", {}))
        for k, v in data.get("bigrams", {}).items():
            self.bigrams[k] = defaultdict(float, v)
        self.destiny = data.get("destiny", [0.0]*8)
        self.trauma = data.get("trauma", 0.0)
        self.total_steps = data.get("total_steps", 0)
        self.chambers = data.get("chambers", self.chambers)


# ═══════════════════════════════════════════════════════════════
# EXTRACT + RUN
# ═══════════════════════════════════════════════════════════════

def extract_key(text):
    words = [w for w in text.lower().split() if len(w) > 1 and w not in STOP]
    if not words:
        parts = text.lower().split()
        return parts[0] if parts else "silence"
    words.sort(key=lambda w: (-len(w), 0))
    return words[0]


def find_seed(key):
    if key in VOCAB_SET:
        return key
    best, best_score = None, -1
    for w in VOCAB:
        score = 0
        if w in key or key in w:
            score = 3
        for i in range(min(len(w), len(key))):
            if w[i] == key[i]:
                score += 0.5
            else:
                break
        if score > best_score:
            best_score, best = score, w
    if best and best_score > 0:
        return best
    return VOCAB[random.randint(0, 199)]


def run_chain(field, text):
    key = extract_key(text)
    seed = find_seed(key)
    prophesied = field.make_prophecy(seed)

    input_words = set(text.lower().split())
    forbidden = set(input_words) | {seed}

    chain = [seed]
    prev = seed
    total_debt = 0.0

    print(f"\n  destined: {prophesied}")
    print(f"\n  {seed}")

    for step in range(STEPS):
        field.update_chambers(step)
        field.prophecy_age += 1

        word = field.select_word(chain, prev, step, forbidden)
        chain.append(word)
        forbidden.add(word)

        field.update_cooc(prev, word)
        field.update_bigram(prev, word)
        widx = VOCAB_IDX.get(word, -1)
        if widx >= 0:
            cat = word_category(widx)
            field.destiny[cat] = 0.3 + 0.7 * field.destiny[cat]

        if word != field.prophecy_target:
            total_debt += 0.1 * math.log(1 + field.prophecy_age)
        else:
            total_debt = max(0, total_debt - 1)

        if step > 7:
            field.trauma = min(1, field.trauma + 0.1)
        field.trauma *= 0.97

        for ctx in chain[-4:]:
            field.update_cooc(ctx, word)

        marker = "  *" if step == STEPS - 1 else "  "
        print(f"{marker}{word}")
        prev = word

    field.total_steps += STEPS
    fulfilled = field.prophecy_target in chain

    cats = set()
    for w in chain:
        idx = VOCAB_IDX.get(w, -1)
        if idx >= 0:
            cats.add(word_category(idx))

    print(f"\n  debt {total_debt:.2f} · resonance {len(field.cooc)/100:.2f} · "
          f"drift {len(cats)}/8 · prophecy {'fulfilled' if fulfilled else 'unfulfilled'}")
    return chain


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    field = DarioField()
    save_path = None
    load_path = None
    text = None

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--save" and i+1 < len(args):
            save_path = args[i+1]; i += 2
        elif args[i] == "--load" and i+1 < len(args):
            load_path = args[i+1]; i += 2
        else:
            text = " ".join(args[i:]); break

    if load_path and os.path.exists(load_path):
        field.load(load_path)
        print(f"  field loaded: {len(field.cooc)} connections")

    print()
    print("  penelope — 1984 words, 12 steps, Dario Equation")
    print("  by Arianna Method")
    print()

    if text:
        run_chain(field, text)
    else:
        while True:
            try:
                text = input("  > ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not text:
                continue
            run_chain(field, text)

    if save_path:
        field.save(save_path)
        print(f"\n  field saved: {len(field.cooc)} connections")


if __name__ == "__main__":
    main()
