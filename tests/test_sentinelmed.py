import random
import re

import pandas as pd
import pytest

from sentinelmed import lexicons as L
from sentinelmed import synth
from sentinelmed.classifier import Confirmer
from sentinelmed.pipeline import SentinelMed, Vault
from sentinelmed.rules import curp_is_valid, find_structured, nss_is_valid
from sentinelmed.textutil import norm
from sentinelmed.training import add_corrections, load_corrections


def labels(text):
    return {(f.label, f.text) for f in find_structured(text)}


# ---------------------------------------------------------------- rules

def test_curp_check_digit_accepts_generated_and_rejects_tampered():
    rng = random.Random(1)
    for r in synth.make_records(50, seed=1):
        curp = r["curp"]
        assert curp_is_valid(curp)
        wrong = curp[:-1] + str((int(curp[-1]) + 1) % 10)
        assert not curp_is_valid(wrong)
    assert rng  # generator is deterministic for a seed
    assert synth.make_records(3, seed=7)[0]["curp"] == synth.make_records(3, seed=7)[0]["curp"]


def test_curp_found_in_lowercase_and_spaced_forms():
    curp = synth.make_records(1, seed=3)[0]["curp"]
    spaced = f"{curp[:4]} {curp[4:10]} {curp[10:]}"
    assert ("CURP", curp.lower()) in labels(f"curp {curp.lower()}.")
    assert ("CURP", spaced) in labels(f"CURP: {spaced}")


def test_nss_luhn():
    nss = synth.make_records(1, seed=4)[0]["nss"]
    assert nss_is_valid(nss)
    assert not nss_is_valid(nss[:-1] + str((int(nss[-1]) + 5) % 10))


@pytest.mark.parametrize("phone", [
    "686 123 4567", "686-123-4567", "(686) 123-4567", "6861234567", "686 123 45 67", "+52 686 123 4567",
    "55 1234 5678", "686.123.4567", "(664)8110240", "664 1 97 90 67", "01 653 8065699", "+52 1 656 365 8996",
])
def test_phone_formats(phone):
    assert ("PHONE", phone) in labels(f"Tel. {phone} por la tarde")


def test_phone_extension_is_part_of_the_phone():
    assert ("PHONE", "442 957 7317 ext. 16") in labels("oficina 442 957 7317 ext. 16.")


def test_numbers_that_are_not_phones():
    found = labels("TA 120/80 mmHg, glucosa 126.5 mg/dL, peso 80 kg, 12/03/2024, c/12 h, C.P. 21000")
    assert not {lab for lab, _ in found} & {"PHONE", "NSS"}


@pytest.mark.parametrize("text,date", [
    ("Fecha de nacimiento: 15/03/1972.", "15/03/1972"), ("Nació el 3 de mayo de 1990.", "3 de mayo de 1990"),
    ("F. nac.: 02/01/2009", "02/01/2009"), ("Fecha nac. 08/01/98", "08/01/98"), ("FN 1944-02-22", "1944-02-22"),
])
def test_birth_dates(text, date):
    assert ("BIRTHDATE", date) in labels(text)


def test_visit_dates_are_left_alone():
    assert not labels("Próxima cita: 12/01/2027. Laboratorios del 4 de octubre de 2026.")


# ---------------------------------------------------------------- language layer (no spaCy, no classifier)

@pytest.fixture(scope="module")
def plain():
    return SentinelMed(use_ner=False, use_classifier=False)


def masked(sm, text):
    return sm.mask_text(text)[0]


def test_names_roles_and_consistency(plain):
    out = masked(plain, "Paciente Juan Carlos Pérez López acude con su esposa María Hernández. "
                        "El Sr. Pérez refiere dolor. Atendió: Dra. Laura Ibarra.")
    assert "Juan" not in out and "María" not in out and "Ibarra" not in out
    assert out.count("[PATIENT_1]") == 2          # full name and "Sr. Pérez" are the same person
    assert "[RELATIVE_1]" in out and "[DOCTOR_1]" in out


def test_all_caps_and_lowercase_notes(plain):
    assert "JUAN" not in masked(plain, "PACIENTE JUAN PEREZ LOPEZ INGRESA A URGENCIAS POR FIEBRE.")
    assert "juan" not in masked(plain, "px juan perez refiere dolor desde ayer.")


def test_unseen_names_through_context_and_shape(plain):
    out = masked(plain, "Identificación: OLACHEA LIZÁRRAGA GRISELDA, 81 años. Efraín, su hijo, administra los "
                        "medicamentos. Ingresa Hortensia Tirado procedente de IMSS.")
    for name in ("OLACHEA", "GRISELDA", "Efraín", "Hortensia", "Tirado"):
        assert name not in out
    assert "IMSS" in out


def test_addresses(plain):
    out = masked(plain, "Vive en Calle Benito Juárez #123, Col. Centro, C.P. 21000, Mexicali, B.C. Tel. 686 123 4567.")
    assert out == "Vive en [ADDRESS_1] Tel. [PHONE_1]."
    out = masked(plain, "Domicilio conocido en Mz. 48 Lt. 22, Fracc. Burócrata. Fractura cerrada de tibia.")
    assert "Mz." not in out and "Fractura cerrada de tibia" in out


def test_clinical_content_survives(plain):
    text = "Se inicia metformina 850 mg cada 12 horas. Glucosa en ayuno 180 mg/dL, HbA1c 8.1%. TA 130/85 mmHg."
    assert masked(plain, text) == text


def test_same_value_same_placeholder_across_formats(plain):
    v = Vault()
    a, _ = plain.mask_text("Tel. 686 123 4567", v, "1")
    b, _ = plain.mask_text("Cel: (686) 123-4567", v, "2")
    assert a.split()[-1] == b.split()[-1] == "[PHONE_1]"


def test_denylist():
    sm = SentinelMed(use_ner=False, use_classifier=False, denylist=["Rancho El Mezquite"])
    assert "Mezquite" not in masked(sm, "Trabaja en Rancho El Mezquite desde 2019.")


# ---------------------------------------------------------------- vocabulary splits used by the evaluation

def test_held_out_names_are_outside_the_detector_list():
    known = L.FIRST_NAMES | L.SURNAMES
    for group in (synth.MALE, synth.FEMALE, synth.SURNAME):
        assert not [w for w in group["E"].split() if norm(w) in known]
    assert not [w for w in synth.DRUG["E"].split() if norm(w) in L.DRUGS]


# ---------------------------------------------------------------- classifier, gate, table, learning loop

class _AlwaysZero:
    """A stand-in model that calls every candidate a non-identifier."""

    def predict_proba(self, X):
        import numpy as np
        return np.array([[1.0, 0.0]] * len(X))


def test_gate_no_reason_no_rejection():
    sm = SentinelMed(use_ner=False, confirmer=Confirmer(_AlwaysZero()))
    text = "Se explica el plan a Zenaida Cauich. Se inicia Metformina 850 mg. Folio 2026110412."
    out, findings = sm.mask_text(text)
    by = {f.text: f for f in findings}
    assert by["Zenaida Cauich"].decision == "review" and "Zenaida" not in out       # masked, flagged
    assert by["2026110412"].decision == "skip" and "2026110412" in out               # rejected with a reason
    assert by["2026110412"].reason
    assert "Metformina 850 mg" in out


@pytest.fixture(scope="module")
def trained():
    sm = SentinelMed(use_ner=False)
    if sm.confirmer is None:
        pytest.skip("no trained model; run `sentinelmed train`")
    return sm


def test_table_end_to_end_leaves_no_structured_identifier(trained):
    recs = synth.make_records(40, seed=99)
    df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in recs])
    out, findings, vault = trained.mask_table(df)
    blob = out.to_csv(index=False)
    for r in recs:
        assert r["curp"] not in blob and r["nss"] not in blob and r["correo"] not in blob
        assert re.sub(r"\D", "", r["telefono"])[-10:] not in re.sub(r"\D", "", blob)
    # numeric and clinical columns are untouched
    assert list(out["glucosa_mg_dl"]) == list(df["glucosa_mg_dl"]) and list(out["diagnostico"]) == list(df["diagnostico"])
    # one patient, one placeholder: the name column and the note agree
    assert out.loc[0, "nombre"].startswith("[PATIENT_")
    assert set(findings["decision"]) <= {"mask", "review", "skip-review", "skip"}
    assert len(vault.key_table()) > 0


def test_reviewer_corrections_are_stored_without_the_identifier(tmp_path, trained):
    df = pd.DataFrame({"nota": ["Se explica el plan a Zenaida Cauich. Recibe Tempra 500 mg."]})
    _, findings, _ = trained.mask_table(df, text_cols=["nota"], id_cols={})
    from sentinelmed.report import write_reports
    write_reports(findings, tmp_path, "t.csv", 1)
    review = pd.read_csv(tmp_path / "review.csv", dtype=str).fillna("")
    review.loc[0, "reviewer_decision"] = "mask"
    review.to_csv(tmp_path / "review.csv", index=False)
    store = tmp_path / "corrections.jsonl"
    assert add_corrections(tmp_path / "review.csv", tmp_path / "findings.jsonl", store) == 1
    X, y = load_corrections(store)
    assert y == [1] and "Zenaida" not in store.read_text(encoding="utf-8")
    assert (tmp_path / "review.html").read_text(encoding="utf-8").startswith("<!doctype html>")
