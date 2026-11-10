"""Synthetic Mexican patient records with gold labels. No real person is described.

Vocabulary is split three ways so the evaluation cannot flatter the tool:

* A - words the detector's own lists contain
* B - words the detector's lists do NOT contain, used for training, so the
      classifier learns that a name outside the list is still a name
* C - held out from training; only the "hard" set uses them. The hard set also
      uses sentence patterns (Q) that never appear in training. It was used
      during development to study errors, so its scores are optimistic.
* D - a second held-out vocabulary with its own sentence patterns (R). Scored
      blind once; that score exposed real weaknesses, so it was then used for a
      second round of fixes and is a development set too.
* E - the final held-out vocabulary and patterns (S), written after all
      development stopped and scored exactly once: the "final" set. Its names
      are checked to be outside the detector's name list.
"""
from __future__ import annotations

import datetime as dt
import random
import re

from .rules import curp_check_digit, luhn_check_digit
from .textutil import deaccent

# ---------------------------------------------------------------- vocabulary
MALE = {
    "A": "José Juan Luis Carlos Jorge Miguel Francisco Jesús Antonio Alejandro Pedro Manuel Ricardo Fernando Roberto "
         "Javier Sergio Daniel Eduardo Rafael Raúl Arturo Enrique Héctor Gerardo Mario Alberto Óscar Víctor Armando "
         "David Andrés Diego Gabriel Hugo Iván Julio Marco Martín Pablo Ramón Rubén Salvador Guillermo Gustavo Felipe "
         "Ernesto Alfredo Ángel Emilio Ignacio Ismael Joel Leonardo Mateo Santiago Sebastián Emiliano Omar Rodrigo "
         "César Adrián Agustín Tomás Vicente Abel Benjamín Cristian Édgar Esteban",
    "B": "Aarón Bernardo Elías Fabián Germán Gilberto Humberto Israel Jaime Lorenzo Marcelo Moisés Noé Octavio "
         "Rogelio Samuel Simón Rolando René Leonel",
    "C": "Uriel Gael Axel Brayan Kevin Jonathan Isaías Ezequiel Saúl Leobardo Rigoberto Heriberto Wenceslao Crisanto "
         "Fidel Ulises Baltazar Efraín Nicanor Teodoro",
    "D": "Anselmo Bonifacio Casimiro Dagoberto Eleazar Florencio Genaro Hipólito Isidro Jacinto Lázaro Macario "
         "Nazario Onésimo Plácido Quirino Remigio Silvestre Tiburcio Valentín",
    "E": "Abundio Baldomero Cayetano Delfino Epifanio Fulgencio Gumaro Hermenegildo Ildefonso Leoncio Melquiades "
         "Nabor Otilio Pánfilo Sabás Tranquilino Venancio Wilebaldo Yahir Zenón Kaleb Dylan Iker Emir Elian Thiago",
}
FEMALE = {
    "A": "María Guadalupe Juana Margarita Verónica Elizabeth Alejandra Leticia Rosa Gabriela Patricia Josefina Teresa "
         "Adriana Ana Martha Silvia Laura Claudia Carmen Alicia Yolanda Araceli Elena Francisca Gloria Isabel Lucía "
         "Sofía Valeria Fernanda Daniela Mariana Paola Karla Diana Norma Irma Sandra Mónica Rocío Susana Beatriz "
         "Blanca Cecilia Cristina Dolores Esperanza Estela Eva Graciela Lourdes Luz Maricela Olga Raquel Rebeca Reyna "
         "Rosario Socorro Angélica Brenda Carolina Erika Julia Lorena Natalia Ximena Regina Camila Andrea",
    "B": "Abigail Alondra Amparo Bertha Carla Elvira Estefanía Jazmín Liliana Marisol Miriam Nancy Ofelia Pamela "
         "Tania Vanessa Viviana Yesenia Zulema Maricruz",
    "C": "Itzel Yaretzi Citlali Nayeli Xóchitl Anahí Zyanya Dulce Perla Abril Renata Romina Fátima Aitana Ivanna "
         "Maribel Noemí Griselda Hortensia Eugenia",
    "D": "Adelaida Benita Candelaria Delfina Eulalia Felícitas Genoveva Herminia Imelda Jacinta Leonor Matilde "
         "Nicolasa Obdulia Petra Refugio Sabina Tomasa Úrsula Zenaida",
    "E": "Agripina Basilia Crescencia Dionisia Eustolia Fidelia Gudelia Hilaria Irais Jovita Librada Macrina "
         "Nereida Otilia Remedios Teodora Xiomara Yatziri Zoila Aranza Danna Kimberly Yamileth Nahomi Evelyn Brisa",
}
SURNAME = {
    "A": "Hernández García Martínez López González Pérez Rodríguez Sánchez Ramírez Cruz Flores Gómez Morales Vázquez "
         "Reyes Jiménez Torres Díaz Gutiérrez Ruiz Mendoza Aguilar Ortiz Moreno Castillo Romero Álvarez Méndez Chávez "
         "Rivera Juárez Ramos Domínguez Herrera Medina Castro Vargas Guzmán Velázquez Rojas Contreras Salazar Luna "
         "Ortega Guerrero Estrada Bautista Cortés Soto Alvarado Espinoza Lara Ávila Ríos Cervantes Silva Delgado Vega "
         "Márquez Sandoval Carrillo Fernández León Mejía Solís Rosas Ibarra Valenzuela Campos Santos Camacho Navarro "
         "Peña Maldonado Rosales Acosta Miranda Trejo Valdez Nava Cabrera Fuentes Zamora Pacheco Núñez Cárdenas Molina "
         "Robles Ayala Salinas Padilla Rubio Bravo Villegas Montoya Ochoa Zavala Figueroa Gallegos Meza Cano Orozco "
         "Paredes Rangel Tapia Valencia Villanueva Zúñiga Arellano Beltrán Calderón Duarte Esquivel Franco Galindo "
         "Lozano Macías Olvera Quintero",
    "B": "Aguirre Alarcón Bermúdez Bustamante Carranza Cisneros Corona Dávila Escobar Gámez Garza Granados Lugo "
         "Madrigal Mora Murillo Ojeda Palacios Ponce Quezada Rentería Rincón Saldaña Serrano Tovar Treviño Urías "
         "Venegas Villarreal Zamudio Leyva Montes Cota Burgos Angulo Barajas Carmona Elizondo Fierro Gaxiola",
    "C": "Xicoténcatl Tzompantzi Cuevas Barragán Echeverría Iñiguez Uribe Yáñez Zepeda Quiroz Arredondo Bañuelos "
         "Covarrubias Villalobos Sepúlveda Monroy Ledesma Huerta Gastélum Félix Osuna Lizárraga Bojórquez Armenta "
         "Cázares Tirado Verdugo Higuera Peralta Olachea Canul Pech Dzul Chan May Poot Tun Ek Cocom Uc",
    "D": "Abarca Balderas Cuéllar Dorantes Escamilla Fonseca Godínez Hinojosa Irigoyen Jaramillo Landeros Mondragón "
         "Negrete Oropeza Pantoja Quiñones Reséndiz Sotelo Tejeda Ugalde Vallejo Zaragoza Yépez Olmos Nájera Mireles "
         "Lemus Jasso Illescas Hurtado Tzuc Cauich Balam Noh Caamal",
    "E": "Ahumada Bocanegra Camarena Dzib Escalera Farfán Gurrola Huitrón Izaguirre Jáuregui Kantún Larios Machuca "
         "Nolasco Ocaña Paniagua Quijano Rebolledo Sagredo Talamantes Uscanga Villafuerte Xool Yam Zatarain Cupul "
         "Tuz Chablé Coyotl Tecuapetla Ixtepan Zamarripa Olguín Mancilla Ibáñez",
}
COMPOUND_SURNAMES = ["De la Cruz", "De León", "Del Río", "De la Torre", "De los Santos", "De la Rosa"]
DRUG = {
    "A": "Metformina Losartán Enalapril Captopril Amlodipino Atorvastatina Omeprazol Paracetamol Ibuprofeno Naproxeno "
         "Diclofenaco Ketorolaco Amoxicilina Ampicilina Ciprofloxacino Azitromicina Ceftriaxona Insulina Glibenclamida "
         "Salbutamol Prednisona Dexametasona Loratadina Ranitidina Furosemida Hidroclorotiazida Metoprolol Propranolol "
         "Clonazepam Sertralina Fluoxetina Levotiroxina Warfarina Aspirina Tramadol Metamizol Clindamicina "
         "Albendazol Metronidazol Pravastatina Telmisartán Sitagliptina Carbamazepina Butilhioscina Ambroxol Nifedipino",
    "B": "Glimepirida Pioglitazona Candesartán Irbesartán Carvedilol Lansoprazol Cefuroxima Claritromicina Fluconazol "
         "Ketoconazol Betametasona Hidrocortisona Cetirizina Diazepam Alprazolam Paroxetina Venlafaxina Risperidona "
         "Haloperidol Alopurinol",
    "C": "Empagliflozina Dapagliflozina Rosuvastatina Valsartán Bisoprolol Pantoprazol Esomeprazol Levofloxacino "
         "Cefalexina Montelukast Budesonida Escitalopram Quetiapina Pregabalina Gabapentina Celecoxib Apixabán "
         "Rivaroxabán Linagliptina Espironolactona",
    "D": "Tamsulosina Finasterida Colchicina Mesalazina Loperamida Ondansetrón Metoclopramida Tiamazol Nistatina "
         "Aciclovir Oseltamivir Ivermectina Glucophage Tempra Tylex Lasix Neomelubrina Buscapina Januvia Micardis",
    "E": "Dimenhidrinato Clorfenamina Dextrometorfano Benzonatato Trimebutina Pinaverio Senósidos Lactulosa "
         "Bezafibrato Ezetimiba Vildagliptina Dulaglutida Nitroglicerina Isosorbida Digoxina Amiodarona Riopan Dolac "
         "Flanax Sedalmerck",
}
EPONYM = {
    "A": ["enfermedad de Parkinson", "enfermedad de Alzheimer", "síndrome de Down", "síndrome de Cushing",
          "linfoma de Hodgkin", "tiroiditis de Hashimoto", "enfermedad de Graves", "enfermedad de Crohn",
          "enfermedad de Addison", "signo de Murphy positivo", "dolor en punto de McBurney", "signo de Blumberg negativo",
          "escala de Glasgow de 15", "signo de Babinski ausente", "maniobra de Valsalva", "signo de Giordano positivo",
          "signo de Homans negativo", "prueba de Romberg negativa", "enfermedad de Chagas", "parálisis de Bell",
          "fenómeno de Raynaud", "enfermedad de Kawasaki", "enfermedad de Wilson", "síndrome de Marfan",
          "síndrome de Turner", "síndrome de Sjögren", "ultrasonido Doppler", "estudio Holter de 24 horas",
          "sonda Foley", "tinción de Gram"],
    "B": ["síndrome de Asperger", "enfermedad de Hirschsprung", "sarcoma de Kaposi", "enfermedad de Lyme",
          "mal de Pott", "síndrome de Reiter", "enfermedad de Still", "granulomatosis de Wegener",
          "maniobra de Barlow", "maniobra de Ortolani", "signo de Lasègue positivo", "enfermedad de Ménière"],
    "C": ["síndrome de Tourette", "enfermedad de Huntington", "esófago de Barrett", "enfermedad de Paget",
          "enfermedad de Behçet", "arteritis de Takayasu", "encefalopatía de Wernicke", "contractura de Dupuytren",
          "signo de Rovsing positivo", "signo de Tinel positivo", "prueba de Phalen negativa", "síndrome de Korsakoff"],
    "D": ["síndrome de Stevens-Johnson", "tetralogía de Fallot", "fractura de Colles", "hernia de Spiegel",
          "quiste de Baker", "nódulos de Heberden", "púrpura de Henoch-Schönlein", "anemia de Fanconi",
          "angina de Prinzmetal", "pie de Charcot", "triada de Virchow", "puntaje de Framingham",
          "clasificación de Wagner", "criterios de Duke", "reflejo de Moro", "enfermedad de Peyronie"],
    "E": ["síndrome de Wolff-Parkinson-White", "enfermedad de Whipple", "signo de Chvostek", "signo de Trousseau",
          "maniobra de Heimlich", "escala de Braden", "escala de Norton", "puntaje de Wells",
          "clasificación de Forrest", "enfermedad de Von Willebrand", "anillo de Kayser-Fleischer",
          "tumor de Wilms", "síndrome de Ehlers-Danlos", "triada de Charcot"],
}
INSTITUTION = {
    "std": ["Hospital General de Mexicali", "IMSS", "ISSSTE", "ISSSTECALI", "Cruz Roja", "Centro de Salud Urbano",
            "UMF 28", "Hospital Ángeles", "Hospital Juárez", "Hospital Civil de Guadalajara", "Hospital Almater",
            "Clínica San José", "Hospital Infantil Federico Gómez"],
    "hard": ["Instituto Nacional de Cardiología Ignacio Chávez", "Clínica Santa María", "Sanatorio San Francisco",
             "Hospital General Dr. Manuel Gea González", "Laboratorios Chopo", "Hospital Materno Infantil",
             "Centro Médico Nacional La Raza", "Hospital Regional Lic. Adolfo López Mateos"],
    "blind": ["Hospital Regional de Alta Especialidad", "Clínica Hospital Constitución", "Hospital Christus Muguerza",
              "Fundación Teletón", "Hospital San Javier", "Centro Médico Siglo XXI",
              "Hospital Dr. Rafael Pascacio Gamboa", "DIF municipal", "Farmacias Similares", "Hospital Star Médica"],
    "final": ["Hospital General Regional 1", "Clínica del Noroeste", "Hospital Hispano Americano",
              "Centro de Salud Rural Disperso", "Hospital de la Mujer", "CAPASITS", "Hospital Militar Regional",
              "Sanatorio Lourdes", "Clínica Dr. José Castro Villagrana", "Cruz Verde"],
}
STREET_NAME = {
    "std": ["Reforma", "Benito Juárez", "Lázaro Cárdenas", "Independencia", "Hidalgo", "Morelos", "Zaragoza",
            "5 de Mayo", "16 de Septiembre", "20 de Noviembre", "Río Colorado", "Insurgentes", "Revolución",
            "Constitución", "Álvaro Obregón", "Venustiano Carranza", "Emiliano Zapata", "Niños Héroes", "Tercera",
            "Novena", "De los Pinos", "Las Palmas", "Del Sol", "Gómez Morín", "Anáhuac", "Colón", "Justo Sierra", "11"],
    "hard": ["Mariano Escobedo", "Xicoténcatl", "Río Presidio", "Lerdo de Tejada", "Ignacio Allende", "De la Patria",
             "Circunvalación", "Guillermo Prieto", "18 de Marzo", "Churubusco", "Quetzalcóatl", "Sor Juana"],
    "blind": ["Paseo de los Héroes", "Vicente Guerrero", "Aquiles Serdán", "Plan de Ayala", "Ignacio Zaragoza",
              "Sonora", "Jalisco", "Tercera Sur", "Diagonal Reforma", "Batalla de Puebla", "Niño Artillero",
              "Héroes de Nacozari", "Adolfo Ruiz Cortines", "Río Mocorito"],
    "final": ["Río Culiacán", "Michoacán", "Francisco Sarabia", "Josefa Ortiz de Domínguez", "Miguel Alemán", "Uxmal",
              "Pípila", "1 de Mayo", "Constituyentes", "Cuitláhuac", "Leona Vicario", "Tlaxcala"],
}
COLONIA_NAME = {
    "std": ["Centro", "Nueva", "Industrial", "Pueblo Nuevo", "Cuauhtémoc", "Los Pinos", "Villa Verde",
            "Jardines del Lago", "Valle Dorado", "San Marcos", "Santa Teresa", "Lomas Altas", "Independencia",
            "Bellavista", "Esperanza", "Libertad", "Las Fuentes", "González Ortega", "Villas del Rey", "Hidalgo"],
    "hard": ["Carbajal", "Orizaba", "Xochimilco", "Robledo", "Wisteria", "Praderas del Sol", "Conjunto Urbano Esperanza",
             "Aurora", "Santa Isabel", "Hacienda Dorada"],
    "blind": ["Burócrata", "Solidaridad", "Emiliano Zapata", "Lomas de Chapultepec", "El Mirador", "Santa Fe",
              "Guadalupe Victoria", "Ampliación Benito Juárez", "Nueva Tijuana", "Popular 89"],
    "final": ["Obrera", "Progreso", "Ex Ejido Coahuila", "Santa Cecilia", "Loma Linda", "Nacionalista", "Calafia",
              "Rivera Campestre", "Villas de la República", "27 de Septiembre"],
}
STREET_TYPE_FINAL = ["Calle", "Av.", "Blvd.", "Calz.", "Privada", "Avenida", "Calle", "Av.", "Blvr.", "C."]
STREET_TYPE_BLIND = ["Calle", "Av.", "Blvd.", "Cda.", "Carr.", "Circuito", "Retorno", "Priv.", "Calle", "Avenida"]
STREET_TYPE = ["Calle", "Calle", "Av.", "Avenida", "Blvd.", "Calz.", "Privada", "Callejón", "Prol.", "Andador"]
CITY = [("Mexicali", "B.C."), ("Mexicali", "Baja California"), ("Tijuana", "B.C."), ("Ensenada", "Baja California"),
        ("Guadalajara", "Jalisco"), ("Monterrey", "Nuevo León"), ("Hermosillo", "Sonora"), ("Culiacán", "Sinaloa"),
        ("Puebla", "Puebla"), ("Mérida", "Yucatán")]
AREA3 = ["686", "664", "646", "665", "653", "662", "667", "614", "656", "222", "442", "999"]
AREA2 = ["55", "33", "81"]
STATE_CODES = ["BC", "SR", "SL", "JC", "NL", "DF", "PL", "YN", "CH", "BS", "MC", "VZ"]
KIN = {"M": ["esposo", "hijo", "padre", "hermano", "abuelo", "tío", "nieto", "sobrino", "papá"],
       "F": ["esposa", "hija", "madre", "hermana", "abuela", "tía", "nieta", "sobrina", "mamá"]}
SYMPTOM = ["dolor abdominal", "cefalea", "fiebre", "tos productiva", "mareo", "disnea de esfuerzo", "dolor torácico",
           "náusea y vómito", "poliuria y polidipsia", "dolor lumbar", "fatiga", "edema de miembros inferiores"]
DX = ["diabetes mellitus tipo 2", "hipertensión arterial sistémica", "dislipidemia", "obesidad grado I", "asma",
      "hipotiroidismo", "infección de vías urinarias", "gastritis crónica", "enfermedad renal crónica", "EPOC",
      "artritis reumatoide", "insuficiencia cardiaca", "migraña", "ansiedad generalizada"]
DOSE = ["500 mg", "850 mg", "20 mg", "50 mg", "10 mg", "100 mg", "5 mg", "40 mg", "1 g", "25 mg", "10 UI", "2.5 mg"]
MONTH = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre",
         "noviembre", "diciembre"]
AMBIGUOUS_PHRASE = {
    "std": ["Paciente delgado, consciente y orientado.", "Dolores articulares de predominio matutino.",
            "Campos pulmonares bien ventilados.", "Cruz Roja realiza el traslado.", "Paciente moreno, bien hidratado.",
            "Refiere dolores en ambas rodillas.", "Tez morena, cabello cano.", "Paciente rubio, sin datos de alarma."],
    "hard": ["Campos pulmonares con estertores basales.", "Paciente franco con su consumo de alcohol.",
             "Luna llena facial por uso de esteroides.", "Dolores musculares generalizados.",
             "Cursa con fiebre desde julio.", "Paciente bravo a la exploración, poco cooperador."],
    "blind": ["Campos pulmonares limpios.", "Paciente con cabello cano y tez blanca.",
              "Luz pupilar con respuesta adecuada.", "Marco temporal de los síntomas: 3 semanas.",
              "Dolores de cabeza frecuentes.", "Paciente con ánimo franco y cooperador.",
              "Buena esperanza de recuperación.", "Se brinda consuelo y contención emocional a la familia."],
    "final": ["Rosa de bengala negativa.", "Mesa quirúrgica disponible.", "Paciente leal a su tratamiento.",
              "Cadena de frío respetada para la vacuna.", "Dolores torácicos opresivos.",
              "Niño sano, esquema de vacunación completo.", "Corona dental fracturada.",
              "Sierra de Juárez como lugar de origen."],
}

# ---------------------------------------------------------------- sentence patterns
P = {  # training patterns
    "intro": [
        "Paciente {patient_full}, {sexword} de {age} años, acude a consulta por {symptom}.",
        "{patient_short} acude a control de {dx}.",
        "Px {patient_full}, {age} años, con antecedente de {dx}.",
        "Nombre: {patient_full}. Edad: {age} años.",
        "{patient_inv} ingresa a urgencias por {symptom}.",
        "Se presenta {title} {patient_sur} refiriendo {symptom} de {n} días de evolución.",
        "{Sexword} de {age} años, de nombre {patient_full}, con diagnóstico de {dx}.",
        "Paciente {sexword} de {age} años con {dx}.",
        "Acude {patient_first} a seguimiento por {dx}.",
    ],
    "family": [
        "Acude acompañad{o} de su {kin} {relative_full}, quien refiere {symptom} desde hace {n} días.",
        "Su {kin}, {relative_full}, es su cuidador principal.",
        "Vive con su {kin} {relative_first} en {address}.",
        "Antecedentes heredofamiliares: {kin} con {dx}.",
        "Padre finado por {dx}.",
        "Madre viva con {dx}.",
        "En caso de emergencia llamar a {relative_full} ({kin}) al {rel_phone}.",
        "Su {kin} {relative_first} refiere que no ha tomado su medicamento.",
    ],
    "clinical": [
        "{Drug} {dose} cada {h} horas por {n} días.",
        "Se inicia {drug} {dose} y se ajusta {drug2}.",
        "Diagnóstico: {dx}.",
        "Antecedente de {eponym}.",
        "{Eponym} en estudio.",
        "Glucosa en ayuno {glu} mg/dL, HbA1c {hba}%.",
        "TA {sys}/{dia} mmHg, FC {fc} lpm, peso {kg} kg.",
        "Referid{o} del {institution} con folio {folio}.",
        "Orden de laboratorio {lab_order}.",
        "{ambiguous}",
        "Continúa con {drug} {dose}; niega alergias.",
        "Se envía a {institution} para valoración.",
    ],
    "admin": [
        "Teléfono de contacto: {phone}.",
        "Cel. {phone}, correo {email}.",
        "Tel: {phone}.",
        "Domicilio: {address}.",
        "Vive en {address}.",
        "CURP {curp}, NSS {nss}.",
        "CURP: {curp}.",
        "Afiliación IMSS: {nss}.",
        "NSS {nss}.",
        "RFC {rfc}.",
        "Fecha de nacimiento: {dob}.",
        "Nació el {dob}.",
        "Atendió: {dr} {doctor}.",
        "Se interconsulta con {el_dr} {doctor}.",
        "Elaboró: {dr} {doctor}, cédula profesional {cedula}.",
        "Próxima cita: {visit_date}.",
        "Correo electrónico: {email}.",
    ],
}
Q = {  # patterns that only the hard test set uses
    "intro": [
        "Se trata de {patient_full}, quien cursa con {dx} desde hace {n} años.",
        "{patient_first} refiere mejoría tras iniciar {drug}.",
        "Ingresa {patient_short} procedente de {institution}.",
        "{title} {patient_sur} regresa por persistencia de {symptom}.",
        "Identificación: {patient_inv}, {age} años.",
        "Hoy se valora a {patient_full} en consulta externa.",
    ],
    "family": [
        "Informante: {relative_full} ({kin} del paciente).",
        "{La_kin} del paciente, de nombre {relative_full}, comenta que presenta {symptom}.",
        "Reside en {address} junto a su {kin}.",
        "{relative_first}, su {kin}, administra los medicamentos.",
        "Contacto alterno: {relative_full}, {rel_phone}.",
        "Abuela materna con {eponym}.",
    ],
    "clinical": [
        "Recibe {drug} {dose} c/{h} h; se suspende {drug2}.",
        "Antecedente de {eponym} diagnosticado en {institution}.",
        "Número de reporte {folio}; muestra {lab_order}.",
        "{Drug} y {drug2} sin cambios.",
        "Exploración: {ambiguous}",
        "Laboratorios del {visit_date}: creatinina {cr} mg/dL.",
    ],
    "admin": [
        "Se localiza en {address}; tel. {phone}.",
        "Clave única: {curp}.",
        "Número de seguridad social {nss}.",
        "Valorad{o} por {el_dr} {doctor} del servicio de medicina interna.",
        "Se contacta vía WhatsApp al {phone}.",
        "Envía resultados a {email}.",
        "F. nac.: {dob}.",
        "Domicilio particular: {address_plain}.",
        "Médico tratante: {doctor}.",
        "Derechohabiente {nss}.",
    ],
}
R = {  # the blind set: written after development, scored once
    "intro": [
        "{Sexword} de {age} años. Nombre del paciente: {patient_full}.",
        "{patient_full} ({age} años) es traíd{o} por {symptom}.",
        "Se recibe a {patient_short} en el servicio de urgencias.",
        "{patient_first} llega caminando, tranquil{o}.",
        "Consulta de primera vez para {title} {patient_sur}.",
        "{patient_inv_comma} — {age} años — {dx}.",
        "Paciente conocid{o} del servicio, {patient_full}, con {dx}.",
        "El día de hoy acudió {patient_short} por {symptom}.",
    ],
    "family": [
        "Lo acompaña {relative_full}, {kin}.",
        "Datos proporcionados por {relative_full}, quien es su {kin}.",
        "{Kin} ({relative_first}) refiere que el paciente no duerme bien.",
        "Responsable legal: {relative_full}; parentesco: {kin}.",
        "Se explica el plan a {relative_first}, {kin} del paciente.",
        "Vive sol{o}; {kin} en {city}.",
        "Antecedente familiar de {eponym} por rama materna.",
        "Teléfono de su {kin}: {rel_phone}.",
    ],
    "clinical": [
        "Tratamiento actual: {drug} {dose} VO c/{h} h.",
        "{Drug} suspendida por intolerancia; se cambia a {drug2}.",
        "Originari{o} de {city}, residente de {city}.",
        "Probable {eponym}; se solicita valoración.",
        "Trasladad{o} desde {institution} en ambulancia.",
        "Biometría hemática: Hb {hb} g/dL, leucocitos {leu}.",
        "No. de solicitud {folio}, clave de estudio {lab_order}.",
        "{ambiguous}",
        "IMC {imc}, SatO2 {sat}%.",
    ],
    "admin": [
        "Tel. casa {phone}; oficina {phone2}.",
        "Puede localizarse al {phone} después de las 5 pm.",
        "Dirección registrada: {address}.",
        "Habita en {address} desde hace {n} años.",
        "Domicilio conocido en {address_mz}.",
        "Registro CURP {curp}.",
        "No. de afiliación {nss}.",
        "Seguro social: {nss}.",
        "Mail: {email}.",
        "Fecha nac. {dob}.",
        "Nacid{o} el {dob} en {city}.",
        "Nota elaborada por {doctor}, R2 de medicina familiar.",
        "Vo. Bo. {dr} {doctor}.",
        "Interconsulta a nutrición con la Lic. {doctor}.",
        "Enf. {doctor} aplica medicamento.",
        "RFC del paciente {rfc}.",
        "Firma: {doctor}.",
    ],
}
S = {  # the final set: written after all development stopped, scored exactly once
    "intro": [
        "{patient_full} es {un} paciente de {age} años que consulta por {symptom}.",
        "Motivo de consulta: {symptom}. Paciente: {patient_short}.",
        "Se atiende a {title} {patient_sur}, de {age} años.",
        "{patient_first}, de {age} años, vino por {symptom}.",
        "Ficha: {patient_inv}. Sexo: {sexword}.",
        "Hoy revisamos a {patient_short}; continúa con {dx}.",
        "Paciente de nombre {patient_full} que acude por primera vez.",
        "{patient_full}; {age} años; {dx}.",
    ],
    "family": [
        "Vino con {relative_full}, que es su {kin}.",
        "El contacto de emergencia es {relative_full}, tel. {rel_phone}.",
        "{relative_first} ({kin}) se queda como acompañante.",
        "Su {kin} se llama {relative_full}.",
        "Quien l{o} cuida es su {kin}, {relative_first}.",
        "Familiar a cargo: {relative_full}.",
        "{Kin} con antecedente de {eponym}.",
        "Hablar con {relative_first} si no contesta.",
    ],
    "clinical": [
        "Toma {drug} {dose} por las mañanas.",
        "Se receta {drug} y {drug2}.",
        "{Drug} le cayó mal; se retira.",
        "Dx presuntivo: {eponym}.",
        "Viene referid{o} de {institution}.",
        "Presión {sys}/{dia}, pulso {fc}.",
        "Estudio número {lab_order}; recibo {folio}.",
        "{ambiguous}",
        "Nació en {city} y trabaja en {city}.",
    ],
    "admin": [
        "Su número es {phone}.",
        "Marcar al {phone} por la tarde.",
        "Radica en {address}.",
        "Dirección: {address}.",
        "Vive por {address_plain}.",
        "CURP del paciente: {curp}.",
        "IMSS {nss}.",
        "Número de seguro {nss}.",
        "Su correo es {email}.",
        "Nacimiento: {dob}.",
        "FN {dob}.",
        "{El_dr} {doctor} l{o} valoró ayer.",
        "Revisó: {doctor} (médico pasante).",
        "Atiende {dr} {doctor}.",
        "Trabajo social: Lic. {doctor}.",
        "Clave RFC {rfc}.",
        "Enviar copia a {email}.",
    ],
}
GOLD = {"patient_inv_comma": "NAME", "phone2": "PHONE", "address_mz": "ADDRESS", "patient_full": "NAME", "patient_short": "NAME", "patient_first": "NAME", "patient_inv": "NAME",
        "patient_sur": "NAME", "relative_full": "NAME", "relative_first": "NAME", "doctor": "NAME",
        "phone": "PHONE", "rel_phone": "PHONE", "address": "ADDRESS", "address_plain": "ADDRESS", "curp": "CURP",
        "nss": "NSS", "rfc": "RFC", "email": "EMAIL", "dob": "BIRTHDATE"}

_PARTICLES = {"DE", "LA", "LAS", "LOS", "DEL", "Y"}


def _w(s: str) -> list[str]:
    return s.split()


# ---------------------------------------------------------------- identifier builders


def _letters(s: str) -> str:
    return deaccent(s.upper().replace("Ñ", "X")).replace("-", "")


def _main_word(s: str) -> str:
    words = [w for w in _letters(s).split() if w not in _PARTICLES]
    return words[0] if words else "X"


def make_curp(first: list[str], pat: str, mat: str, sex: str, dob: dt.date, state: str, rng: random.Random) -> str:
    p, m = _main_word(pat), _main_word(mat)
    names = [_letters(n) for n in first if _letters(n) not in _PARTICLES]
    name = names[1] if len(names) > 1 and names[0] in {"JOSE", "MARIA", "MA", "J"} else names[0]

    def vowel(s):
        return next((c for c in s[1:] if c in "AEIOU"), "X")

    def cons(s):
        return next((c for c in s[1:] if c.isalpha() and c not in "AEIOU"), "X")

    homo = str(rng.randint(0, 9)) if dob.year < 2000 else rng.choice("ABCDEFGH")
    body = f"{p[0]}{vowel(p)}{m[0]}{name[0]}{dob:%y%m%d}{sex}{state}{cons(p)}{cons(m)}{cons(name)}{homo}"
    return body + curp_check_digit(body)


def make_nss(dob: dt.date, rng: random.Random) -> str:
    reg = min(dob.year + rng.randint(16, 30), 2025)
    body = f"{rng.randint(1, 97):02d}{reg % 100:02d}{dob.year % 100:02d}{rng.randint(0, 9999):04d}"
    return body + luhn_check_digit(body)


def make_phone(rng: random.Random, level: str = "std") -> str:
    if rng.random() < 0.25:
        a, rest = rng.choice(AREA2), f"{rng.randint(10000000, 99999999)}"
        forms = [f"{a} {rest[:4]} {rest[4:]}", f"{a}-{rest[:4]}-{rest[4:]}", f"{a}{rest}", f"+52 {a} {rest[:4]} {rest[4:]}"]
    else:
        a, rest = rng.choice(AREA3), f"{rng.randint(1000000, 9999999)}"
        forms = [f"{a} {rest[:3]} {rest[3:]}", f"{a}-{rest[:3]}-{rest[3:]}", f"({a}) {rest[:3]}-{rest[3:]}", f"{a}{rest}",
                 f"{a} {rest[:3]} {rest[3:5]} {rest[5:]}", f"+52 {a} {rest[:3]} {rest[3:]}", f"{a}.{rest[:3]}.{rest[3:]}"]
        if level == "hard":
            forms += [f"044 {a} {rest[:3]} {rest[3:]}", f"01 {a} {rest}", f"({a}){rest}", f"+52 1 {a} {rest[:3]} {rest[3:]}"]
        if level == "final":
            forms += [f"({a}){rest[:3]}-{rest[3:]}", f"{a} {rest}", f"52 {a} {rest[:3]} {rest[3:]}",
                      f"{a}-{rest[:3]}-{rest[3:5]}-{rest[5:]}"]
        if level == "blind":
            forms += [f"{a} {rest[:3]} {rest[3:]} ext. {rng.randint(10, 99)}", f"{a}-{rest}", f"+52 ({a}) {rest[:3]} {rest[3:]}",
                      f"{a} {rest[0]} {rest[1:3]} {rest[3:5]} {rest[5:]}"]
    return rng.choice(forms)


def make_address(rng: random.Random, level: str = "std", plain: bool = False, mz: bool = False) -> str:
    grp = level if level != "std" and rng.random() < 0.7 else "std"
    if mz:   # lot-and-block addresses, common in newer developments
        return (f"Mz. {rng.randint(1, 60)} Lt. {rng.randint(1, 40)}, "
                f"{rng.choice(['Col.', 'Fracc.', 'Colonia'])} {rng.choice(COLONIA_NAME[grp])}")
    number = rng.choice([f"#{rng.randint(1, 3999)}", f"No. {rng.randint(1, 999)}", f"{rng.randint(1, 2999)}",
                         f"{rng.randint(100, 1999)}-{rng.choice('ABC')}", f"# {rng.randint(1, 999)}"])
    street = rng.choice(STREET_NAME[grp])
    types = {"blind": STREET_TYPE_BLIND, "final": STREET_TYPE_FINAL}.get(level, STREET_TYPE)
    s = f"{street} {number}" if plain else f"{rng.choice(types)} {street} {number}"
    if rng.random() < 0.1:
        s += rng.choice([f" Int. {rng.randint(1, 20)}", f", Depto. {rng.randint(1, 9)}{rng.choice('AB')}"])
    if plain or rng.random() < 0.75:
        s += rng.choice([", ", " "]) + rng.choice(["Col.", "Colonia", "Fracc.", "Fraccionamiento"]) + " " + rng.choice(COLONIA_NAME[grp])
    if rng.random() < 0.35:
        s += rng.choice([", C.P. ", ", CP ", " C.P. "]) + f"{rng.randint(20000, 99999)}"
    if rng.random() < 0.45:
        city, state = rng.choice(CITY)
        s += f", {city}" + (f", {state}" if rng.random() < 0.6 else "")
    return s


def make_dob_text(d: dt.date, rng: random.Random) -> str:
    return rng.choice([f"{d:%d/%m/%Y}", f"{d:%d-%m-%Y}", f"{d.day} de {MONTH[d.month - 1]} de {d.year}",
                       f"{d:%Y-%m-%d}", f"{d:%d/%m/%y}"])


# ---------------------------------------------------------------- people


class Person:
    def __init__(self, rng: random.Random, level: str, sex: str | None = None, pat: str | None = None):
        self.sex = sex or rng.choice("HM")
        pool = MALE if self.sex == "H" else FEMALE

        def pick(groups: dict) -> str:
            if level != "std" and rng.random() < 0.7:
                return rng.choice(_w(groups[_HELD_OUT[level]]))
            return rng.choice(_w(groups["A"]) if rng.random() < 0.75 else _w(groups["B"]))

        self.first = [pick(pool)]
        if rng.random() < 0.3:
            second = pick(pool)
            if second != self.first[0]:
                self.first.append(second)
        elif self.sex == "M" and rng.random() < 0.06:
            self.first = ["María", rng.choice(["de los Ángeles", "del Carmen", "del Rosario", "de Lourdes"])]
        self.pat = pat or (rng.choice(COMPOUND_SURNAMES) if rng.random() < 0.04 else pick(SURNAME))
        self.mat = pick(SURNAME)
        self.dob = dt.date(rng.randint(1940, 2015), rng.randint(1, 12), rng.randint(1, 28))
        self.state = rng.choice(STATE_CODES)

    @property
    def given(self) -> str:
        return " ".join(self.first)

    @property
    def full(self) -> str:
        return f"{self.given} {self.pat} {self.mat}"

    @property
    def short(self) -> str:
        return f"{self.first[0]} {self.pat}"

    @property
    def inverted(self) -> str:
        return f"{self.pat} {self.mat} {self.given}".upper()

    @property
    def inverted_comma(self) -> str:
        return f"{self.pat} {self.mat}, {self.given}".upper()


# ---------------------------------------------------------------- record generation

_SLOT = re.compile(r"\{(\w+)\}")


def _fill(template: str, values: dict) -> tuple[str, list[tuple[int, int, str]]]:
    out, spans, pos, length = [], [], 0, 0
    for m in _SLOT.finditer(template):
        lit = template[pos:m.start()]
        val = values[m.group(1)]()
        val = val if isinstance(val, str) else str(val)
        out.append(lit)
        length += len(lit)
        if m.group(1) in GOLD:
            spans.append((length, length + len(val), GOLD[m.group(1)]))
        out.append(val)
        length += len(val)
        pos = m.end()
        if val.endswith(".") and template[pos:pos + 1] == ".":   # "..., B.C." + "." -> one period
            pos += 1
    out.append(template[pos:])
    return "".join(out), spans


def _strip_accents_keep_n(s: str) -> str:
    return "".join(c if c in "ñÑ" else deaccent(c) for c in s)


_HELD_OUT = {"hard": "C", "blind": "D", "final": "E"}


def make_record(rng: random.Random, idx: int, level: str = "std") -> dict:
    hard = level != "std"
    p = Person(rng, level)
    rel_sex = rng.choice("HM")
    rel = Person(rng, level, sex=rel_sex, pat=p.pat if rng.random() < 0.4 else None)
    doc = Person(rng, level)
    kin = rng.choice(KIN["M" if rel_sex == "H" else "F"])
    age = max(1, 2026 - p.dob.year)
    curp = make_curp(p.first, p.pat, p.mat, p.sex, p.dob, p.state, rng)
    nss = make_nss(p.dob, rng)
    phone, rel_phone = make_phone(rng, level), make_phone(rng, level)
    address = make_address(rng, level)
    email = (f"{deaccent(p.first[0]).lower()}{rng.choice(['.', '_', ''])}{deaccent(_main_word(p.pat)).lower()}"
             f"{rng.choice(['', str(rng.randint(1, 99))])}@"
             f"{rng.choice(['gmail.com', 'hotmail.com', 'outlook.com', 'yahoo.com.mx', 'live.com.mx', 'uabc.edu.mx'])}")
    fem = p.sex == "M"

    def grp(d: dict):
        if hard and rng.random() < 0.7:
            return d[_HELD_OUT[level]]
        return d["A"] if rng.random() < 0.7 else d["B"]

    def drug(cap: bool = False):
        d = rng.choice(_w(grp(DRUG)))
        return d if cap or rng.random() < 0.4 else d.lower()

    def eponym(cap: bool = False):
        e = rng.choice(grp(EPONYM))
        return e[0].upper() + e[1:] if cap else e

    def nss_text():
        return rng.choice([nss, nss, f"{nss[:2]} {nss[2:4]} {nss[4:6]} {nss[6:10]} {nss[10]}", f"{nss[:10]}-{nss[10]}"])

    def curp_text():
        r = rng.random()
        return curp.lower() if r < 0.1 else (f"{curp[:4]} {curp[4:10]} {curp[10:]}" if r < 0.15 else curp)

    doc_title = ("Dra." if doc.sex == "M" else "Dr.")
    inst = INSTITUTION[level] if hard and rng.random() < 0.6 else INSTITUTION["std"]
    amb = AMBIGUOUS_PHRASE[level] if hard and rng.random() < 0.6 else AMBIGUOUS_PHRASE["std"]
    values = {
        "patient_full": lambda: p.full, "patient_short": lambda: p.short, "patient_first": lambda: p.first[0],
        "patient_inv": lambda: p.inverted, "patient_sur": lambda: p.pat,
        "patient_inv_comma": lambda: p.inverted_comma, "Kin": lambda: kin.capitalize(),
        "city": lambda: rng.choice(CITY)[0], "phone2": lambda: make_phone(rng, level),
        "address_mz": lambda: make_address(rng, level, mz=True),
        "hb": lambda: f"{rng.uniform(8, 17):.1f}", "leu": lambda: rng.randint(3500, 15000),
        "imc": lambda: f"{rng.uniform(18, 42):.1f}", "sat": lambda: rng.randint(86, 99),
        "title": lambda: rng.choice(["Sra.", "la Sra.", "Doña"] if fem else ["Sr.", "el Sr.", "Don"]),
        "relative_full": lambda: rng.choice([rel.short, rel.full]), "relative_first": lambda: rel.first[0],
        "kin": lambda: kin, "La_kin": lambda: ("El " if rel_sex == "H" else "La ") + kin,
        "doctor": lambda: rng.choice([doc.short, doc.full, f"{doc.pat} {doc.mat}"]),
        "dr": lambda: rng.choice([doc_title, doc_title.rstrip(".")]),
        "el_dr": lambda: ("la " if doc.sex == "M" else "el ") + doc_title,
        "El_dr": lambda: ("La " if doc.sex == "M" else "El ") + doc_title, "un": lambda: "una" if fem else "un",
        "phone": lambda: phone, "rel_phone": lambda: rel_phone, "address": lambda: address,
        "address_plain": lambda: make_address(rng, level, plain=True),
        "curp": curp_text, "nss": nss_text, "rfc": lambda: curp[:10] + "".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ0123456789") for _ in range(3)),
        "email": lambda: email, "dob": lambda: make_dob_text(p.dob, rng),
        "age": lambda: age, "sexword": lambda: "femenino" if fem else "masculino",
        "Sexword": lambda: "Femenino" if fem else "Masculino", "o": lambda: "a" if fem else "o",
        "symptom": lambda: rng.choice(SYMPTOM), "dx": lambda: rng.choice(DX),
        "drug": drug, "drug2": drug, "Drug": lambda: drug(True), "dose": lambda: rng.choice(DOSE),
        "eponym": eponym, "Eponym": lambda: eponym(True), "institution": lambda: rng.choice(inst),
        "ambiguous": lambda: rng.choice(amb),
        "n": lambda: rng.randint(2, 15), "h": lambda: rng.choice([6, 8, 12, 24]),
        "glu": lambda: rng.randint(80, 320), "hba": lambda: f"{rng.uniform(5.5, 11):.1f}",
        "sys": lambda: rng.randint(100, 170), "dia": lambda: rng.randint(60, 105), "fc": lambda: rng.randint(58, 110),
        "kg": lambda: rng.randint(48, 120), "cr": lambda: f"{rng.uniform(0.6, 2.4):.1f}",
        "folio": lambda: f"{rng.randint(2019, 2026)}{rng.randint(0, 999999):06d}",          # 10 digits, like a phone
        "lab_order": lambda: f"{rng.randint(10**10, 10**11 - 1)}",                           # 11 digits, like an NSS
        "cedula": lambda: rng.randint(1000000, 12999999),
        "visit_date": lambda: f"{rng.randint(1, 28)} de {rng.choice(MONTH)} de 2026",
    }

    def pool(kind: str) -> list[str]:
        if level in ("blind", "final"):
            return (R if level == "blind" else S)[kind]
        return Q[kind] if hard and rng.random() < 0.7 else P[kind]

    sentences = [rng.choice(pool("intro"))]
    body = ([rng.choice(pool("family")) for _ in range(rng.randint(0, 2))]
            + [rng.choice(pool("clinical")) for _ in range(rng.randint(1, 3))]
            + [rng.choice(pool("admin")) for _ in range(rng.randint(2, 4))])
    rng.shuffle(body)
    note, spans = "", []
    for tpl in sentences + body:
        text, sp = _fill(tpl, values)
        if note:
            note += " "
        spans += [(s + len(note), e + len(note), lab) for s, e, lab in sp]
        note += text

    r = rng.random()
    if r < 0.15:
        note = note.upper()
    elif r < 0.22:
        note = note.lower()
    if rng.random() < 0.25:
        note = _strip_accents_keep_n(note)

    return {
        "id": f"EXP-{idx:05d}",
        "nombre": p.inverted if rng.random() < 0.3 else p.full,
        "curp": curp, "nss": nss, "telefono": phone, "correo": email, "direccion": address,
        "fecha_nacimiento": f"{p.dob:%d/%m/%Y}", "sexo": "F" if fem else "M", "edad": age,
        "diagnostico": rng.choice(DX), "medicamento": rng.choice(_w(DRUG["A"])),
        "glucosa_mg_dl": rng.randint(80, 320), "nota": note,
        "_spans": sorted(spans),
        "_patient_tokens": sorted({w for w in re.findall(r"[^\W\d_]+", deaccent(p.full).lower())
                                   if w not in {"de", "del", "la", "las", "los"}}),
    }


def make_records(n: int, seed: int = 0, hard: bool = False, level: str | None = None) -> list[dict]:
    """level: 'std' (default), 'hard' and 'blind' (development) or 'final' (scored once)."""
    rng = random.Random(seed)
    level = level or ("hard" if hard else "std")
    return [make_record(rng, i + 1, level) for i in range(n)]
