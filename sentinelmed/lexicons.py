"""Word lists used by the detector.

Everything is stored normalised (lowercase, no accents) so lookups work on
ALL-CAPS notes and on notes typed without accents.

These lists are deliberately small. They are a starting vocabulary, not a
census: the context rules, the NER model and the classifier exist because no
list of names is ever complete.
"""
from __future__ import annotations

from pathlib import Path

from .textutil import norm

_DATA = Path(__file__).resolve().parent / "data"


def _file(name: str) -> frozenset[str]:
    path = _DATA / name
    return frozenset(norm(w) for w in path.read_text(encoding="utf-8").split()) if path.exists() else frozenset()


def _s(words: str) -> frozenset[str]:
    return frozenset(norm(w) for w in words.split())


FIRST_NAMES = _s("""
José Juan Luis Carlos Jorge Miguel Francisco Jesús Antonio Alejandro Pedro Manuel Ricardo Fernando
Roberto Javier Sergio Daniel Eduardo Rafael Raúl Arturo Enrique Héctor Gerardo Mario Alberto Óscar
Víctor Armando David Andrés Diego Gabriel Hugo Iván Julio Marco Martín Pablo Ramón Rubén Salvador
Guillermo Gustavo Felipe Ernesto Alfredo Ángel Emilio Ignacio Ismael Joel Leonardo Mateo Santiago
Sebastián Emiliano Omar Rodrigo César Adrián Agustín Tomás Vicente Abel Benjamín Cristian Édgar
Esteban María Guadalupe Juana Margarita Verónica Elizabeth Alejandra Leticia Rosa Gabriela Patricia
Josefina Teresa Adriana Ana Martha Silvia Laura Claudia Carmen Alicia Yolanda Araceli Elena
Francisca Gloria Isabel Lucía Sofía Valeria Fernanda Daniela Mariana Paola Karla Diana Norma Irma
Sandra Mónica Rocío Susana Beatriz Blanca Cecilia Cristina Dolores Esperanza Estela Eva Graciela
Lourdes Luz Maricela Olga Raquel Rebeca Reyna Rosario Socorro Angélica Brenda Carolina Erika Julia
Lorena Natalia Ximena Regina Camila Andrea Ángeles Pilar Consuelo Mercedes Soledad
""")

FIRST_NAMES |= _file("first_names_es_mx.txt")     # larger list, from the Faker project (MIT licence)

SURNAMES = _s("""
Hernández García Martínez López González Pérez Rodríguez Sánchez Ramírez Cruz Flores Gómez Morales
Vázquez Reyes Jiménez Torres Díaz Gutiérrez Ruiz Mendoza Aguilar Ortiz Moreno Castillo Romero
Álvarez Méndez Chávez Rivera Juárez Ramos Domínguez Herrera Medina Castro Vargas Guzmán Velázquez
Rojas Contreras Salazar Luna Ortega Guerrero Estrada Bautista Cortés Soto Alvarado Espinoza Lara
Ávila Ríos Cervantes Silva Delgado Vega Márquez Sandoval Carrillo Fernández León Mejía Solís Rosas
Ibarra Valenzuela Campos Santos Camacho Navarro Peña Maldonado Rosales Acosta Miranda Trejo Valdez
Nava Cabrera Fuentes Zamora Pacheco Núñez Cárdenas Molina Robles Ayala Salinas Padilla Rubio Bravo
Villegas Montoya Ochoa Zavala Figueroa Gallegos Meza Cano Orozco Paredes Rangel Tapia Valencia
Villanueva Zúñiga Arellano Beltrán Calderón Duarte Esquivel Franco Galindo Lozano Macías Olvera
Quintero
""")

SURNAMES |= _file("surnames_es_mx.txt")

# Names that are also ordinary Spanish words ("paciente delgado", "dolores
# abdominales", "Cruz Roja"). They stay in the lists above, but the classifier
# is told about the ambiguity.
AMBIGUOUS = _s("""
Delgado Moreno Rubio Bravo Blanca Cano Franco Luna Cruz León Flores Ríos Campos Rosas Peña Vega Nava
Lara Santos Reyes Rosales Fuentes Paredes Torres Castillo Guerrero Dolores Luz Rosa Ángel Socorro
Esperanza Rosario Consuelo Pilar Soledad Mercedes Ángeles Reyna Santiago Valencia Salinas Estrada
Robles Medina Molina Montoya Julio Marco Salvador Gloria Estela
Alma Alba Aurora Clara Dulce Linda Perla Reina Gracia Caridad Concepción Natividad Trinidad Victoria
Violeta Esmeralda Cristal Modesto Amador Blanco Bueno Caballero Cadena Caldera Canales Casas Casillas
Ceja Corona Coronado Corral Corrales Fuente Guardado Guerra Iglesias Jurado Leal Limón Lira Llamas
Madera Madrid Mares Mata Mesa Mora Muro Naranjo Nieves Niño Olivo Partida Paz Puente Raya Regalado
Rico Río Rojo Rosado Salas Segura Sierra Toro Torre Vaca Valle Valles Vela Vera Villa Sevilla Toledo
Zaragoza Hidalgo Colón Mercado Prado Parra Pulido Preciado Zapata Arenas Armas Arroyo Barrera Barrios
Carrera Cordero Crespo Espino Lucero Maya Montes Palomo Piña Porras Posada Bustos Becerra Mota Roque
""")

NAME_PARTICLES = _s("de del la las los")

DRUGS = _s("""
Metformina Losartán Enalapril Captopril Amlodipino Atorvastatina Omeprazol Paracetamol Ibuprofeno
Naproxeno Diclofenaco Ketorolaco Amoxicilina Ampicilina Ciprofloxacino Azitromicina Ceftriaxona
Insulina Glibenclamida Salbutamol Prednisona Dexametasona Loratadina Ranitidina Furosemida
Hidroclorotiazida Metoprolol Propranolol Clonazepam Sertralina Fluoxetina Levotiroxina Warfarina
Aspirina Tramadol Metamizol Clindamicina Nitrofurantoína Albendazol Metronidazol Pravastatina
Telmisartán Sitagliptina Carbamazepina Butilhioscina Ambroxol Glargina Lispro Nifedipino
""")

# Eponyms and other medical words that look like surnames.
MEDICAL_TERMS = _s("""
Parkinson Alzheimer Down Cushing Hodgkin Hashimoto Graves Crohn Addison Guillain Barré Murphy
McBurney Blumberg Glasgow Apgar Babinski Valsalva Kernig Brudzinski Giordano Homans Romberg Lasègue
Chagas Bell Raynaud Kawasaki Wilson Marfan Turner Sjögren Ménière Papanicolaou Doppler Holter Foley
Kocher Gram Koch
""")

INSTITUTION_WORDS = _s("""
hospital clínica clinica centro instituto unidad umf imss issste isssteca issstecali seguro cruz
roja sanatorio dispensario farmacia laboratorio laboratorios secretaría universidad cetys uabc dif
""")

# Units that follow a drug name, never a person's name.
UNITS = _s("mg mcg g ml l ui u tabletas tableta tabs tab cápsulas capsulas gotas ampolleta ampolletas")

# Words before a number that say "this is paperwork, not a phone or NSS".
NUMBER_PAPERWORK = _s("""
folio orden muestra reporte lote receta cédula cedula factura expediente registro referencia código
codigo clave pedido ticket solicitud estudio trámite póliza contrato cuenta matrícula serie recibo
comprobante
""")
PHONE_TRIGGERS = _s("tel teléfono telefono cel celular móvil movil whatsapp contacto llamar marcar")
NSS_TRIGGERS = _s("nss imss afiliación afiliacion seguridad social seguro derechohabiente")

KIN_TERMS = _s("""
esposa esposo hijo hija madre padre mamá papá hermano hermana abuela abuelo tío tía nieto nieta
sobrino sobrina pareja cónyuge suegra suegro cuñado cuñada primo prima novio novia tutor tutora
""")
DOCTOR_TITLES = _s("dr dra doctor doctora enf enfermera enfermero lic médico médica tratante")
PERSON_TITLES = _s("sr sra srita señor señora don doña")
PATIENT_TRIGGERS = _s("paciente pac px nombre")
OTHER_NAME_TRIGGERS = _s("informante cuidador cuidadora acompañante familiar responsable atendió elaboró revisó llamado llamada")
NAME_TRIGGERS = KIN_TERMS | DOCTOR_TITLES | PERSON_TITLES | PATIENT_TRIGGERS | OTHER_NAME_TRIGGERS

# Verbs whose subject is a person: the words just before them may be a name
# ("Itzel refiere mejoría", "KEVIN OLACHEA ACUDE").
PERSON_VERBS = _s("""
refiere acude comenta niega presenta ingresa menciona dice reporta regresa cursa solicita informa
asiste llega vive reside administra acompaña explica
""")
# Verbs whose object is a person, reached through "a" or "de": "se valora a X", "se trata de X".
OBJECT_VERBS = _s("trata valora llamar llama cita contacta avisa entrevista atiende informar avisar")
# Field labels that are followed by a person's name.
PERSON_LABELS = _s("contacto identificación responsable titular beneficiario beneficiaria alterno emergencia")
ABBREVIATIONS = _s("dr dra sr sra srita enf lic pac px")
# Words that introduce a disease, sign or test named after someone.
EPONYM_HEADS = _s("""
enfermedad síndrome signo escala maniobra linfoma tiroiditis prueba punto parálisis fenómeno mal
sarcoma tinción sonda ultrasonido estudio test criterios clasificación índice reflejo tumor bacilo
""")

# Words that are never part of a person's name. Used to end a name that
# follows a trigger when capitalisation gives no clue (ALL-CAPS notes).
STOPWORDS = _s("""
a al algo ante antes aquí así aún bajo bien cada casi como con contra cual cuando de del desde donde
dos durante el ella ellos en entre era es esa ese eso esta este esto fue ha hace hacia han hasta hay
la las le les lo los más me mi muy ni no nos o otra otro para pero poco por que quien se según sin
sobre su sus también tiene tras tres un una uno unos y ya si será son junto
paciente px pac masculino femenino hombre mujer edad años año meses mes días día horas hora semanas
semana acude acuden refiere refieren presenta presentan ingresa ingreso egresa egreso niega cursa
cursando inicia inicio continúa suspende ajusta indica solicita valorado valorada atendió atendido
atendida elaboró revisó consulta control cita urgencias servicio medicina interna general familiar
hospital clínica centro salud unidad instituto nacional seguro social diagnóstico dx tratamiento tx
plan nota evolución antecedente antecedentes heredofamiliares personales patológicos alergias negadas
negados dolor fiebre tos cefalea náusea vómito mareo disnea fatiga diarrea glucosa presión
arterial peso talla temperatura frecuencia cardiaca respiratoria ta fc fr mg ml kg cm tabletas tableta
cápsulas vía oral intravenosa dosis teléfono tel cel celular contacto correo email domicilio dirección
vive reside nombre curp nss rfc afiliación folio orden laboratorio expediente número fecha nacimiento
nació próxima signo síndrome enfermedad escala maniobra prueba positivo negativo normal estable
consciente orientado orientada finado finada vivo viva sano sana acompañado acompañada cuidador
cuidadora principal informante emergencia caso llamar localiza envía resultados recibe mejoría trata
diabetes hipertensión mellitus tipo crónica aguda leve moderada severa referido referida derecho
izquierdo abdomen tórax comenta menciona reporta dice explica quien cuyo cuya llamado llamada
materna materno paterna paterno hba imc ekg ecg refiriendo presentando cursando acudiendo
""") | NAME_TRIGGERS | PERSON_VERBS | OBJECT_VERBS | PERSON_LABELS | UNITS

STREET_TYPES = _s("""
calle av ave avenida blvd boulevard bulevar calz calzada priv privada andador callejón cerrada cda
prol prolongación circuito paseo retorno carretera carr diagonal blvr avda
""")
COLONIA_WORDS = _s("col colonia fracc fraccionamiento ejido barrio residencial infonavit")

CITIES = frozenset(norm(c) for c in """
Mexicali|Tijuana|Ensenada|Tecate|Rosarito|Ciudad de México|CDMX|Guadalajara|Monterrey|Puebla|Hermosillo|
Culiacán|Mérida|León|Querétaro|Toluca|Chihuahua|Ciudad Juárez|Saltillo|Morelia|Oaxaca|Veracruz|Cancún|
Tuxtla Gutiérrez|Villahermosa|Aguascalientes|San Luis Potosí|Zapopan|Tlaquepaque|Nezahualcóyotl|Ecatepec|
Tlalpan|Coyoacán|Acapulco|Durango|Torreón|Mazatlán|La Paz|Los Mochis|Nogales|San Luis Río Colorado|
Ciudad Obregón|Tepic|Colima|Pachuca|Cuernavaca|Xalapa|Campeche|Chetumal|Zacatecas|San Felipe
""".replace("\n", "").split("|"))

STATES = frozenset(norm(c) for c in """
Aguascalientes|Baja California|Baja California Sur|B.C.|BC|B.C.S.|Campeche|Chiapas|Chihuahua|Coahuila|
Colima|Durango|Guanajuato|Guerrero|Hidalgo|Jalisco|Estado de México|Edo. de México|Michoacán|Morelos|
Nayarit|Nuevo León|N.L.|Oaxaca|Puebla|Querétaro|Quintana Roo|San Luis Potosí|Sinaloa|Sonora|Son.|Tabasco|
Tamaulipas|Tlaxcala|Veracruz|Yucatán|Zacatecas|Jal.|Sin.
""".replace("\n", "").split("|"))

# Street types that are also everyday words ("fractura cerrada", "consulta
# privada"): they only start an address when a number or a colonia follows.
AMBIGUOUS_STREET_TYPES = _s("privada cerrada paseo retorno circuito carretera carr andador prol prolongación")
ADDRESS_TRIGGERS = _s("domicilio dirección direccion vive reside habita localiza radica")
