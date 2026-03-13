/*
 * penelope.c — 1984 words. 12 steps of resonance. Dario Equation.
 *
 * Weightless resonance engine. Not a neural network. A mirror.
 * Co-occurrence IS the weight. Repetition IS learning.
 *
 *   score(w) = B + α·H + β·F + γ·A + T
 *       B = bigram affinity (positional pattern, RRPRAM-like)
 *       H = Hebbian co-occurrence with context
 *       F = prophecy fulfillment pressure
 *       A = destiny attraction (category compass)
 *       T = trauma gravity
 *
 *   cc penelope.c -O2 -lm -o penelope
 *   ./penelope                          # interactive
 *   ./penelope "darkness eats the city" # single chain
 *
 * By Arianna Method. הרזוננס לא נשבר
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <ctype.h>

#define NWORDS   1984
#define STEPS    12
#define MAX_COOC 32768
#define MAX_BIG  16384
#define MAX_CTX  64

/* ═══════════════════════════════════════════════════════════════
 * 1984 WORDS
 * ═══════════════════════════════════════════════════════════════ */

static const char *VOCAB[NWORDS] = {
/* BODY 0-99 */
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
/* NATURE 100-199 */
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
/* EMOTION 200-299 */
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
/* TIME 300-349 */
"moment","instant","second","minute","hour","day","night","week","month","year",
"decade","century","epoch","era","age","past","present","future","memory","tomorrow",
"yesterday","forever","never","always","sometimes","often","seldom","once","twice","origin",
"ending","beginning","duration","interval","pause","wait","rush","delay","haste","eternity",
"cycle","season","spring","summer","autumn","winter","dawn","twilight","midnight","noon",
/* SOCIETY 350-449 */
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
/* ABSTRACT 450-549 */
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
/* ACTION 550-649 */
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
/* MATERIAL 650-749 */
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
/* FOOD 750-799 */
"bread","salt","sugar","honey","milk","butter","cheese","meat","fish","egg",
"grain","rice","wheat","corn","fruit","apple","grape","olive","lemon","pepper",
"wine","water","tea","coffee","broth","soup","stew","feast","crumb","morsel",
"harvest","garden","soil","compost","ferment","yeast","dough","crust","marrow","nectar",
"spice","herb","mint","thyme","sage","garlic","onion","mushroom","berry","kernel",
/* ARCHITECTURE 800-849 */
"house","room","wall","floor","ceiling","door","window","stair","corridor","basement",
"tower","bridge","arch","column","dome","vault","foundation","ruin","temple","altar",
"threshold","passage","labyrinth","maze","chamber","cell","shelter","fortress","prison","garden",
"roof","chimney","hearth","frame","beam","pillar","brick","mortar","tile","glass",
"balcony","terrace","courtyard","gate","fence","path","road","intersection","tunnel","well",
/* RELATIONSHIP 850-929 */
"mother","father","child","daughter","son","sister","brother","family","ancestor","descendant",
"friend","stranger","lover","enemy","neighbor","companion","rival","mentor","student","witness",
"husband","wife","partner","orphan","widow","elder","infant","twin","cousin","godmother",
"promise","oath","vow","contract","alliance","betrayal","reconciliation","farewell","reunion","absence",
"kiss","embrace","handshake","slap","caress","quarrel","conversation","confession","accusation","apology",
"birth","death","marriage","divorce","inheritance","adoption","abandonment","protection","neglect","sacrifice",
"trust","suspicion","loyalty","treachery","devotion","indifference","jealousy","admiration","dependence","autonomy",
"intimacy","distance","connection","isolation","belonging","exile","homecoming","departure","waiting","return",
/* PHILOSOPHY 930-999 */
"consciousness","awareness","perception","sensation","intuition","reason","logic","paradox","dialectic","synthesis",
"freedom","determinism","causation","contingency","necessity","possibility","impossibility","actuality","potential","becoming",
"subject","object","self","other","identity","difference","sameness","change","permanence","flux",
"being","nothingness","existence","essence","phenomena","noumena","appearance","reality","illusion","truth",
"ethics","morality","virtue","vice","good","evil","right","wrong","duty","choice",
"justice","mercy","punishment","reward","guilt","innocence","responsibility","consequence","intention","action",
"language","meaning","sign","reference","representation","interpretation","understanding","misunderstanding","translation","silence",
/* MUSIC 1000-1049 */
"melody","rhythm","chord","pitch","tone","note","bass","treble","octave","harmony",
"dissonance","resonance","vibration","frequency","amplitude","tempo","beat","rest","pause","crescendo",
"murmur","hum","buzz","click","crack","boom","rumble","chime","echo","reverb",
"song","lullaby","anthem","dirge","hymn","ballad","fugue","sonata","requiem","improvisation",
"strum","pluck","strike","bow","mute","sustain","fade","loop","drone","overtone",
/* WEATHER 1050-1099 */
"rain","drizzle","downpour","hail","sleet","blizzard","hurricane","tornado","drought","flood",
"breeze","gale","typhoon","monsoon","frost","thaw","haze","smog","rainbow","mirage",
"erosion","sedimentation","crystallization","evaporation","condensation","precipitation","sublimation","oxidation","combustion","decay",
"magma","lava","quartz","granite","obsidian","chalk","slate","sandstone","limestone","basalt",
"marsh","delta","gorge","ridge","summit","abyss","chasm","rift","fault","crater",
/* RITUAL 1100-1149 */
"prayer","meditation","ritual","ceremony","blessing","curse","oath","vow","pilgrimage","procession",
"offering","sacrifice","communion","baptism","funeral","wedding","coronation","initiation","exile","absolution",
"incense","candle","bell","chant","mantra","psalm","scripture","prophecy","oracle","vision",
"mask","costume","dance","feast","fast","vigil","silence","confession","penance","redemption",
"altar","shrine","temple","tomb","relic","artifact","amulet","talisman","totem","icon",
/* LABOR 1150-1199 */
"harvest","planting","sowing","reaping","threshing","milling","baking","brewing","weaving","spinning",
"carving","sculpting","painting","drawing","writing","printing","binding","stitching","welding","forging",
"mining","drilling","excavation","construction","demolition","repair","restoration","invention","discovery","experiment",
"apprentice","craftsman","artist","engineer","architect","farmer","sailor","miner","healer","scribe",
"workshop","studio","laboratory","field","dock","quarry","furnace","mill","press","loom",
/* GEOMETRY 1200-1249 */
"circle","spiral","line","curve","angle","edge","center","margin","border","frame",
"sphere","cube","pyramid","cylinder","cone","helix","vortex","arc","wave","fractal",
"symmetry","asymmetry","proportion","ratio","scale","dimension","plane","axis","vertex","intersection",
"pattern","grid","lattice","mesh","tessellation","rotation","reflection","translation","dilation","projection",
"surface","volume","area","perimeter","diameter","radius","tangent","normal","parallel","perpendicular",
/* ANIMAL 1250-1299 */
"horse","dog","cat","bird","fish","snake","bear","fox","rabbit","turtle",
"eagle","sparrow","raven","swan","heron","falcon","vulture","pelican","nightingale","lark",
"lion","tiger","elephant","giraffe","hippopotamus","rhinoceros","gorilla","chimpanzee","orangutan","leopard",
"salmon","trout","shark","dolphin","octopus","jellyfish","starfish","seahorse","crab","lobster",
"frog","lizard","crocodile","chameleon","gecko","iguana","newt","toad","salamander","viper",
/* COLOR 1300-1349 */
"red","blue","green","white","black","gray","amber","violet","indigo","scarlet",
"crimson","azure","emerald","ivory","obsidian","silver","golden","copper","rust","ochre",
"bright","dark","transparent","opaque","matte","glossy","rough","smooth","coarse","fine",
"stripe","dot","plaid","solid","gradient","shadow","highlight","contrast","saturation","hue",
"velvet","satin","linen","denim","lace","gauze","burlap","chiffon","tweed","corduroy",
/* TRANSPORT 1350-1399 */
"ship","boat","canoe","raft","anchor","sail","rudder","oar","mast","hull",
"train","rail","station","platform","ticket","journey","passage","crossing","departure","arrival",
"wheel","axle","road","highway","path","trail","bridge","tunnel","gate","crossroad",
"wing","flight","altitude","turbulence","landing","orbit","trajectory","velocity","acceleration","gravity",
"horse","carriage","wagon","cart","sled","bicycle","motorcycle","automobile","truck","ambulance",
/* DOMESTIC 1400-1449 */
"kitchen","bedroom","bathroom","attic","cellar","closet","drawer","shelf","table","chair",
"bed","pillow","blanket","curtain","carpet","lamp","mirror","photograph","vase","clock",
"plate","spoon","knife","fork","cup","pot","pan","kettle","oven","stove",
"soap","towel","broom","bucket","needle","thread","button","zipper","hanger","basket",
"door","window","lock","key","handle","hinge","nail","screw","bolt","hook",
/* COMMUNICATION 1450-1499 */
"letter","envelope","stamp","address","message","telegram","telephone","radio","broadcast","signal",
"newspaper","headline","article","column","editorial","report","announcement","rumor","gossip","testimony",
"ink","pen","pencil","typewriter","keyboard","screen","printer","paper","notebook","diary",
"conversation","dialogue","monologue","debate","argument","negotiation","compromise","ultimatum","declaration","speech",
"translation","interpretation","code","cipher","encryption","decryption","password","signature","seal","authentication",
/* MEDICAL 1500-1549 */
"diagnosis","symptom","treatment","remedy","cure","relapse","recovery","surgery","anesthesia","bandage",
"infection","inflammation","fracture","hemorrhage","allergy","immunity","vaccine","antibiotic","toxin","antidote",
"hospital","clinic","pharmacy","laboratory","ambulance","stretcher","scalpel","syringe","stethoscope","thermometer",
"fever","cough","rash","swelling","numbness","dizziness","insomnia","fatigue","nausea","tremor",
"pulse","pressure","temperature","respiration","circulation","digestion","metabolism","reflex","coordination","balance",
/* COSMIC 1550-1599 */
"universe","galaxy","constellation","planet","asteroid","meteorite","satellite","orbit","void","singularity",
"photon","electron","proton","neutron","atom","molecule","particle","quantum","field","dimension",
"spacetime","relativity","entropy","thermodynamics","radiation","spectrum","wavelength","frequency","amplitude","interference",
"supernova","blackhole","pulsar","quasar","nebula","wormhole","antimatter","darkmatter","redshift","expansion",
"telescope","observatory","mission","launch","countdown","trajectory","reentry","landing","exploration","discovery",
/* BUREAUCRACY 1600-1649 */
"document","form","permit","license","certificate","registration","application","approval","denial","appeal",
"regulation","compliance","violation","penalty","exemption","quota","deadline","protocol","procedure","standard",
"office","desk","file","folder","stamp","signature","receipt","invoice","ledger","archive",
"committee","department","ministry","bureau","agency","institution","organization","corporation","foundation","commission",
"report","audit","review","inspection","evaluation","assessment","benchmark","statistic","data","record",
/* MYTHIC 1650-1699 */
"oracle","prophecy","fate","destiny","curse","blessing","quest","trial","sacrifice","redemption",
"labyrinth","threshold","guardian","shadow","mirror","mask","transformation","metamorphosis","resurrection","apocalypse",
"phoenix","dragon","serpent","sphinx","minotaur","chimera","hydra","golem","specter","wraith",
"underworld","paradise","purgatory","limbo","abyss","eden","babylon","atlantis","olympus","tartarus",
"hero","villain","trickster","sage","fool","maiden","crone","warrior","healer","shapeshifter",
/* TEXTUAL 1700-1749 */
"word","sentence","paragraph","chapter","verse","stanza","line","margin","footnote","epilogue",
"prologue","preface","title","subtitle","dedication","inscription","epitaph","motto","slogan","proverb",
"metaphor","simile","allegory","irony","satire","parody","tragedy","comedy","farce","melodrama",
"narrator","character","protagonist","antagonist","audience","reader","author","critic","editor","translator",
"manuscript","draft","revision","erasure","correction","annotation","citation","reference","index","bibliography",
/* PSYCHOLOGICAL 1750-1799 */
"unconscious","subconscious","conscious","ego","superego","libido","repression","projection","sublimation","transference",
"trauma","complex","fixation","regression","denial","rationalization","displacement","compensation","identification","dissociation",
"archetype","persona","anima","animus","shadow","self","individuation","integration","fragmentation","wholeness",
"attachment","separation","abandonment","dependency","autonomy","codependency","boundary","enmeshment","differentiation","fusion",
"grief","mourning","acceptance","bargaining","anger","depression","recovery","relapse","healing","scarring",
/* FINAL 1800-1983 */
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
};

/* ═══════════════════════════════════════════════════════════════
 * STOPWORDS
 * ═══════════════════════════════════════════════════════════════ */

static const char *STOPS[] = {
"i","me","my","we","our","you","your","he","she","it","they","them","the","a","an",
"and","or","but","in","on","at","to","for","of","is","am","are","was","were","be",
"been","being","have","has","had","do","does","did","will","would","shall","should",
"can","could","may","might","must","not","no","nor","so","if","then","than","that",
"this","these","those","what","which","who","whom","how","when","where","why","all",
"each","every","some","any","few","many","much","more","most","other","another","such",
NULL
};

static int is_stop(const char *w) {
    for (int i = 0; STOPS[i]; i++)
        if (strcmp(w, STOPS[i]) == 0) return 1;
    return 0;
}

static int find_word(const char *w) {
    for (int i = 0; i < NWORDS; i++)
        if (strcmp(VOCAB[i], w) == 0) return i;
    return -1;
}

static int word_category(int idx) {
    if (idx < 100) return 0;
    if (idx < 200) return 1;
    if (idx < 300) return 2;
    if (idx < 350) return 3;
    if (idx < 450) return 4;
    if (idx < 550) return 5;
    if (idx < 650) return 6;
    return 7;
}

/* ═══════════════════════════════════════════════════════════════
 * DARIO FIELD
 * ═══════════════════════════════════════════════════════════════ */

typedef struct { int a, b; float val; } CoocEntry;
typedef struct { int prev, next; float val; } BigramEntry;

static CoocEntry   cooc[MAX_COOC];
static int         cooc_n = 0;
static BigramEntry bigs[MAX_BIG];
static int         big_n = 0;
static float       destiny[8] = {0};
static float       trauma = 0;
static int         prophecy_target = -1;
static int         prophecy_age = 0;
static int         total_steps = 0;

/* chambers */
enum { CH_FEAR=0, CH_LOVE, CH_RAGE, CH_VOID, CH_FLOW, CH_COMPLEX, NCH };
static float chambers[NCH] = {0};
static const float ch_decay[NCH] = {0.95f, 0.95f, 0.93f, 0.96f, 0.94f, 0.97f};

static float randf(void) { return (float)rand() / (float)RAND_MAX; }
static float clampf(float x, float lo, float hi) { return x<lo?lo:(x>hi?hi:x); }

static void cooc_update(int a, int b) {
    if (a > b) { int t=a; a=b; b=t; }
    for (int i = 0; i < cooc_n; i++)
        if (cooc[i].a == a && cooc[i].b == b) { cooc[i].val += 1.0f; return; }
    if (cooc_n < MAX_COOC)
        cooc[cooc_n++] = (CoocEntry){a, b, 1.0f};
}

static float cooc_get(int a, int b) {
    if (a > b) { int t=a; a=b; b=t; }
    for (int i = 0; i < cooc_n; i++)
        if (cooc[i].a == a && cooc[i].b == b) return cooc[i].val;
    return 0;
}

static void bigram_update(int prev, int next) {
    for (int i = 0; i < big_n; i++)
        if (bigs[i].prev == prev && bigs[i].next == next) { bigs[i].val += 1.0f; return; }
    if (big_n < MAX_BIG)
        bigs[big_n++] = (BigramEntry){prev, next, 1.0f};
}

static float bigram_get(int prev, int next) {
    for (int i = 0; i < big_n; i++)
        if (bigs[i].prev == prev && bigs[i].next == next) return bigs[i].val;
    return 0;
}

static float bigram_max(int prev) {
    float mx = 0;
    for (int i = 0; i < big_n; i++)
        if (bigs[i].prev == prev && bigs[i].val > mx) mx = bigs[i].val;
    return mx;
}

static void update_chambers(int step_idx) {
    float depth = (float)step_idx / STEPS;
    int phase = depth < 0.33f ? 0 : (depth < 0.66f ? 1 : 2);
    if (phase == 0) chambers[CH_FLOW] += 0.05f;
    if (phase == 1) chambers[CH_FEAR] += 0.04f;
    if (phase == 2) chambers[CH_VOID] += 0.05f;
    if (depth > 0.75f) chambers[CH_COMPLEX] += 0.03f;
    if (trauma > 0.3f) chambers[CH_RAGE] += 0.04f;

    float K = 0.02f, old[NCH];
    memcpy(old, chambers, sizeof(old));
    for (int i = 0; i < NCH; i++)
        for (int j = 0; j < NCH; j++)
            if (i != j) chambers[i] += K * sinf(old[j] - old[i]);

    for (int i = 0; i < NCH; i++)
        chambers[i] = clampf(chambers[i] * ch_decay[i], 0, 1);
}

/* ═══════════════════════════════════════════════════════════════
 * DARIO EQUATION
 * ═══════════════════════════════════════════════════════════════ */

static float dario_score(int cand, const int *ctx, int ctx_n, int prev, int step) {
    float alpha_mod = 1 + 0.3f*chambers[CH_LOVE] - 0.2f*chambers[CH_RAGE] + 0.1f*chambers[CH_FLOW];
    float beta_mod  = 1 + 0.2f*chambers[CH_FLOW] - 0.3f*chambers[CH_FEAR];
    float gamma_mod = 1 + 0.4f*chambers[CH_VOID] + 0.2f*chambers[CH_COMPLEX];

    /* B: bigram */
    float B = 0;
    if (prev >= 0) {
        float mx = bigram_max(prev);
        B = bigram_get(prev, cand) / (mx + 1);
    }

    /* H: Hebbian */
    float H = 0;
    for (int i = 0; i < ctx_n; i++)
        H += cooc_get(ctx[i], cand) * (1.0f / (ctx_n - i + 1));
    if (H > 1) H = 1;

    /* F: prophecy */
    float F = 0;
    if (prophecy_target >= 0 && cand == prophecy_target)
        F = 0.5f * logf(1.0f + prophecy_age);

    /* A: destiny */
    int cat = word_category(cand);
    float d_max = 0.01f;
    for (int i = 0; i < 8; i++) if (fabsf(destiny[i]) > d_max) d_max = fabsf(destiny[i]);
    float A = destiny[cat] / d_max;

    /* T: trauma */
    float T = 0;
    if (trauma > 0.3f && step > 6) {
        if (cand >= 200 && cand < 300) T = trauma * 1.5f;
        else if (cand >= 450 && cand < 550) T = trauma * 1.2f;
        else if (cand >= 930 && cand < 1000) T = trauma * 1.0f;
    }

    /* wormhole */
    float worm = 0;
    if (randf() < 0.08f && step > 3 && step < 10)
        if (cat == rand() % 8) worm = 0.5f;

    float vibes[] = {1.3f,1.1f,1.0f,0.9f,0.8f,0.7f,0.8f,0.9f,1.0f,0.8f,0.7f,1.4f};
    float vibe = (step < 12) ? vibes[step] : 1.0f;
    float tau = 0.7f + step * 0.08f;

    return (B*8 + alpha_mod*0.3f*H + beta_mod*0.15f*F +
            gamma_mod*0.25f*A + T + worm) * vibe / tau;
}

/* ═══════════════════════════════════════════════════════════════
 * SELECT + CHAIN
 * ═══════════════════════════════════════════════════════════════ */

static int select_word(const int *ctx, int ctx_n, int prev, int step, const int *forbidden, int nforbid) {
    typedef struct { int idx; float score; } Sc;
    static Sc scores[NWORDS];
    int n = 0;

    float noise = 0.3f * (1.0f - (float)step / STEPS);

    for (int w = 0; w < NWORDS; w++) {
        int skip = 0;
        for (int f = 0; f < nforbid; f++) if (forbidden[f] == w) { skip = 1; break; }
        if (skip) continue;
        scores[n++] = (Sc){w, dario_score(w, ctx, ctx_n, prev, step) + randf() * noise};
    }

    /* sort descending */
    for (int i = 0; i < n-1; i++)
        for (int j = i+1; j < n; j++)
            if (scores[j].score > scores[i].score) { Sc t = scores[i]; scores[i] = scores[j]; scores[j] = t; }

    /* top-k=12 sampling */
    int k = n < 12 ? n : 12;
    float sum = 0.001f;
    for (int i = 0; i < k; i++) sum += scores[i].score > 0 ? scores[i].score : 0;
    float r = randf() * sum;
    for (int i = 0; i < k; i++) {
        r -= scores[i].score > 0 ? scores[i].score : 0;
        if (r <= 0) return scores[i].idx;
    }
    return scores[0].idx;
}

static int find_seed(const char *key) {
    int idx = find_word(key);
    if (idx >= 0) return idx;

    int best = -1; float best_score = -1;
    for (int i = 0; i < NWORDS; i++) {
        float score = 0;
        if (strstr(VOCAB[i], key) || strstr(key, VOCAB[i])) score = 3;
        int ml = strlen(VOCAB[i]), kl = strlen(key);
        int mn = ml < kl ? ml : kl;
        for (int j = 0; j < mn; j++) {
            if (VOCAB[i][j] == key[j]) score += 0.5f;
            else break;
        }
        if (score > best_score) { best_score = score; best = i; }
    }
    return (best >= 0 && best_score > 0) ? best : (rand() % 200);
}

static void extract_key(const char *text, char *out, int maxlen) {
    char buf[1024];
    int bi = 0;
    for (int i = 0; text[i] && bi < 1022; i++)
        buf[bi++] = tolower(text[i]);
    buf[bi] = 0;

    char *best = NULL; int best_len = 0;
    char *tok = strtok(buf, " \t\n");
    while (tok) {
        if (strlen(tok) > 1 && !is_stop(tok)) {
            if ((int)strlen(tok) > best_len) { best = tok; best_len = strlen(tok); }
        }
        tok = strtok(NULL, " \t\n");
    }
    if (best) { strncpy(out, best, maxlen-1); out[maxlen-1]=0; }
    else { strncpy(out, "silence", maxlen-1); out[maxlen-1]=0; }
}

static void run_chain(const char *text) {
    char key[64];
    extract_key(text, key, sizeof(key));
    int seed = find_seed(key);

    /* prophecy */
    int deep_cats[] = {2, 5, 7};
    int tcat = deep_cats[rand() % 3];
    int ranges[][2] = {{0,100},{100,200},{200,300},{300,350},{350,450},{450,550},{550,650},{650,1984}};
    prophecy_target = ranges[tcat][0] + rand() % (ranges[tcat][1] - ranges[tcat][0]);
    if (prophecy_target >= NWORDS) prophecy_target = NWORDS - 1;
    prophecy_age = 0;

    printf("\n  destined: %s\n", VOCAB[prophecy_target]);
    printf("\n  %s\n", VOCAB[seed]);

    int chain[STEPS+1], chain_n = 0;
    int forbidden[STEPS+100], nforbid = 0;
    chain[chain_n++] = seed;
    forbidden[nforbid++] = seed;

    int prev = seed;
    float total_debt = 0;
    int fulfilled = 0;

    for (int step = 0; step < STEPS; step++) {
        update_chambers(step);
        prophecy_age++;

        int word = select_word(chain, chain_n, prev, step, forbidden, nforbid);
        chain[chain_n++] = word;
        forbidden[nforbid++] = word;

        cooc_update(prev, word);
        bigram_update(prev, word);
        int cat = word_category(word);
        destiny[cat] = 0.3f + 0.7f * destiny[cat];

        if (word != prophecy_target)
            total_debt += 0.1f * logf(1.0f + prophecy_age);
        else {
            total_debt = total_debt > 1 ? total_debt - 1 : 0;
            fulfilled = 1;
        }

        if (step > 7) trauma = trauma + 0.1f < 1 ? trauma + 0.1f : 1;
        trauma *= 0.97f;

        int start = chain_n > 4 ? chain_n - 4 : 0;
        for (int c = start; c < chain_n; c++)
            cooc_update(chain[c], word);

        if (step == STEPS - 1)
            printf("  *%s\n", VOCAB[word]);
        else
            printf("   %s\n", VOCAB[word]);
        prev = word;
    }

    total_steps += STEPS;

    int cats_seen = 0, cat_flags[8] = {0};
    for (int i = 0; i < chain_n; i++) {
        int c = word_category(chain[i]);
        if (!cat_flags[c]) { cat_flags[c] = 1; cats_seen++; }
    }

    printf("\n  debt %.2f \xc2\xb7 resonance %.2f \xc2\xb7 drift %d/8 \xc2\xb7 prophecy %s\n",
           total_debt, cooc_n / 100.0f, cats_seen,
           fulfilled ? "fulfilled" : "unfulfilled");
}

/* ═══════════════════════════════════════════════════════════════
 * MAIN
 * ═══════════════════════════════════════════════════════════════ */

int main(int argc, char **argv) {
    srand(time(NULL));
    printf("\n  penelope \xe2\x80\x94 1984 words, 12 steps, Dario Equation\n");
    printf("  by Arianna Method\n\n");

    if (argc > 1) {
        char buf[1024] = {0};
        for (int i = 1; i < argc; i++) {
            if (i > 1) strcat(buf, " ");
            strncat(buf, argv[i], sizeof(buf) - strlen(buf) - 2);
        }
        run_chain(buf);
        return 0;
    }

    char line[1024];
    while (1) {
        printf("  > ");
        fflush(stdout);
        if (!fgets(line, sizeof(line), stdin)) break;
        int len = strlen(line);
        while (len > 0 && (line[len-1] == '\n' || line[len-1] == '\r')) line[--len] = 0;
        if (len == 0) continue;
        run_chain(line);
    }

    return 0;
}
