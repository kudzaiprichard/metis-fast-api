"""
Neo4j Graph Database for ML Explainability.

Provides:
- Similar patient case finding (tabular + graph)
- Patient detail lookup by Neo4j ID
- Clinical guidelines retrieval
- Contraindication checking
- Drug interaction warnings
- Background data for SHAP analysis
"""

from neo4j import GraphDatabase
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)


class Neo4jGraphDatabase:

    def __init__(self, uri: str, username: str, password: str):
        self.uri = uri
        self.username = username
        self.password = password
        self.driver = None
        self._connected = False

    # ─── Connection lifecycle ───

    def connect(self) -> bool:
        try:
            self.driver = GraphDatabase.driver(
                self.uri, auth=(self.username, self.password)
            )
            with self.driver.session() as session:
                session.run("RETURN 1")
            self._connected = True
            logger.info("Connected to Neo4j at %s", self.uri)
            return True
        except Exception:
            self._connected = False
            return False

    def close(self) -> None:
        if self.driver:
            self.driver.close()
            self._connected = False
            logger.info("Neo4j connection closed")

    def is_connected(self) -> bool:
        return self._connected and self.driver is not None

    # ─── Similar patients (tabular) ───

    def find_similar_patients(
        self,
        patient_profile: Dict[str, Any],
        limit: int = 5,
        treatment_filter: Optional[str] = None,
        min_similarity: float = 0.5,
    ) -> List[Dict[str, Any]]:
        if not self.is_connected():
            logger.warning("Neo4j not connected for similar patients search")
            return []

        age = patient_profile.get("age", 50)
        hba1c = patient_profile.get("hba1c_baseline", 8.0)
        c_peptide = patient_profile.get("c_peptide", 1.5)
        bmi = patient_profile.get("bmi", 30)
        egfr = patient_profile.get("egfr", 90)
        diabetes_duration = patient_profile.get("diabetes_duration", 5.0)

        comorbidities = []
        comorbidity_map = {
            "hypertension": "Hypertension",
            "ckd": "CKD",
            "cvd": "CVD",
            "nafld": "NAFLD",
            "retinopathy": "Retinopathy",
        }
        for key, name in comorbidity_map.items():
            if patient_profile.get(key, 0) == 1:
                comorbidities.append(name)

        age_group = self._get_age_group(age)
        hba1c_severity = self._get_hba1c_severity(hba1c)
        limit = max(1, min(limit, 20))
        min_similarity = max(0.0, min(min_similarity, 1.0))

        try:
            with self.driver.session() as session:
                treatment_clause = (
                    "AND t.drug_name = $treatment_filter" if treatment_filter else ""
                )

                query = f"""
                MATCH (p:Patient)-[:RECEIVED_TREATMENT]->(t:Treatment)
                WHERE p.age_group = $age_group
                  AND p.hba1c_severity = $hba1c_severity
                  AND abs(p.hba1c_baseline - $hba1c) <= 2.0
                  AND abs(p.c_peptide - $c_peptide) <= 0.5
                  {treatment_clause}

                MATCH (o:Outcome {{patient_id: p.patient_id}})
                OPTIONAL MATCH (p)-[:HAS_CONDITION]->(c:Comorbidity)

                WITH p, t, o, collect(DISTINCT c.condition_name) AS patient_comorbidities,
                     (abs(p.age - $age) / 60.0) +
                     (abs(p.hba1c_baseline - $hba1c) / 7.0) * 2.0 +
                     (abs(p.c_peptide - $c_peptide) / 3.0) * 3.0 +
                     (abs(p.bmi - $bmi) / 20.0) +
                     (abs(p.egfr - $egfr) / 100.0) +
                     (abs(p.diabetes_duration - $diabetes_duration) / 30.0) AS normalized_distance

                WITH p, t, o, patient_comorbidities, normalized_distance,
                     size([x IN patient_comorbidities WHERE x IN $comorbidities]) AS overlap_count,
                     size(patient_comorbidities) + size($comorbidities) -
                     size([x IN patient_comorbidities WHERE x IN $comorbidities]) AS union_count

                WITH p, t, o, patient_comorbidities, normalized_distance,
                     CASE
                       WHEN union_count > 0
                       THEN toFloat(overlap_count) / union_count
                       ELSE CASE WHEN size(patient_comorbidities) = 0 AND size($comorbidities) = 0 THEN 1.0 ELSE 0.0 END
                     END AS comorbidity_similarity

                WITH p, t, o, patient_comorbidities, normalized_distance, comorbidity_similarity,
                     1.0 - (normalized_distance / 6.0) AS clinical_similarity

                WITH p, t, o, patient_comorbidities,
                     clinical_similarity,
                     comorbidity_similarity,
                     (clinical_similarity * 0.7) + (comorbidity_similarity * 0.3) AS similarity_score

                WHERE similarity_score >= $min_similarity
                ORDER BY similarity_score DESC
                LIMIT $limit

                RETURN p.patient_id AS patient_id,
                       p.age AS age,
                       p.gender AS gender,
                       p.ethnicity AS ethnicity,
                       p.hba1c_baseline AS hba1c_baseline,
                       p.c_peptide AS c_peptide,
                       p.bmi AS bmi,
                       p.egfr AS egfr,
                       p.diabetes_duration AS diabetes_duration,
                       p.bp_systolic AS bp_systolic,
                       p.fasting_glucose AS fasting_glucose,
                       patient_comorbidities,
                       t.drug_name AS treatment,
                       t.drug_class AS drug_class,
                       o.hba1c_reduction AS hba1c_reduction,
                       o.hba1c_followup AS hba1c_followup,
                       CASE
                         WHEN o.time_to_target IS NULL OR toString(o.time_to_target) = 'NaN'
                         THEN 'Unknown'
                         ELSE o.time_to_target
                       END AS time_to_target,
                       CASE
                         WHEN o.adverse_events IS NULL OR toString(o.adverse_events) = 'NaN'
                         THEN 'None'
                         ELSE o.adverse_events
                       END AS adverse_events,
                       o.outcome_category AS outcome_category,
                       o.success AS success,
                       similarity_score,
                       clinical_similarity,
                       comorbidity_similarity
                """

                result = session.run(
                    query,
                    age_group=age_group,
                    hba1c_severity=hba1c_severity,
                    age=age,
                    hba1c=hba1c,
                    c_peptide=c_peptide,
                    bmi=bmi,
                    egfr=egfr,
                    diabetes_duration=diabetes_duration,
                    comorbidities=comorbidities,
                    limit=limit,
                    min_similarity=min_similarity,
                    treatment_filter=treatment_filter,
                )

                similar_cases = []
                for record in result:
                    similar_cases.append({
                        "case_id": record["patient_id"],
                        "similarity_score": round(record["similarity_score"], 3),
                        "clinical_similarity": round(record["clinical_similarity"], 3),
                        "comorbidity_similarity": round(record["comorbidity_similarity"], 3),
                        "profile": {
                            "age": int(record["age"]),
                            "gender": record["gender"],
                            "ethnicity": record["ethnicity"],
                            "hba1c_baseline": round(record["hba1c_baseline"], 1),
                            "c_peptide": round(record["c_peptide"], 2),
                            "bmi": round(record["bmi"], 1),
                            "egfr": round(record["egfr"], 1),
                            "diabetes_duration": round(record["diabetes_duration"], 1),
                            "bp_systolic": int(record["bp_systolic"]),
                            "fasting_glucose": round(record["fasting_glucose"], 1),
                        },
                        "comorbidities": record["patient_comorbidities"] or [],
                        "treatment_given": record["treatment"],
                        "drug_class": record["drug_class"],
                        "outcome": {
                            "hba1c_reduction": round(record["hba1c_reduction"], 1),
                            "hba1c_followup": round(record["hba1c_followup"], 1),
                            "time_to_target": record["time_to_target"],
                            "adverse_events": record["adverse_events"],
                            "outcome_category": record["outcome_category"],
                            "success": bool(record["success"]),
                        },
                    })

                logger.info(
                    "Found %d similar patients (filter: %s, min_similarity: %s)",
                    len(similar_cases),
                    treatment_filter or "none",
                    min_similarity,
                )
                return similar_cases

        except Exception as e:
            logger.error("Error finding similar patients: %s", e)
            return []

    # ─── Similar patients (graph) ───

    def find_similar_cases_graph(
        self,
        patient_profile: Dict[str, Any],
        limit: int = 5,
        treatment_filter: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not self.is_connected():
            return {"nodes": [], "edges": [], "metadata": {"error": "Neo4j not connected"}}

        age = patient_profile.get("age", 50)
        hba1c = patient_profile.get("hba1c_baseline", 8.0)
        c_peptide = patient_profile.get("c_peptide", 1.5)
        bmi = patient_profile.get("bmi", 30)

        comorbidities = []
        for condition in ("hypertension", "ckd", "cvd", "nafld", "retinopathy"):
            if patient_profile.get(condition, 0) == 1:
                comorbidities.append(condition.upper())

        age_group = self._get_age_group(age)
        hba1c_severity = self._get_hba1c_severity(hba1c)

        try:
            with self.driver.session() as session:
                treatment_clause = (
                    "AND t.drug_name = $treatment_filter" if treatment_filter else ""
                )

                query = f"""
                MATCH (p:Patient)-[:RECEIVED_TREATMENT]->(t:Treatment)
                WHERE p.age_group = $age_group
                  AND p.hba1c_severity = $hba1c_severity
                  AND abs(p.hba1c_baseline - $hba1c) <= 2.0
                  AND abs(p.c_peptide - $c_peptide) <= 0.5
                  {treatment_clause}

                MATCH (o:Outcome {{patient_id: p.patient_id}})
                OPTIONAL MATCH (p)-[:HAS_CONDITION]->(c:Comorbidity)

                WITH p, t, o, collect(DISTINCT c.condition_name) AS patient_comorbidities,
                     abs(p.age - $age) +
                     abs(p.hba1c_baseline - $hba1c) * 2.0 +
                     abs(p.c_peptide - $c_peptide) * 3.0 AS distance

                WITH p, t, o, patient_comorbidities, distance,
                     size([x IN patient_comorbidities WHERE x IN $comorbidities]) AS overlap_count,
                     size(patient_comorbidities) + size($comorbidities) AS total_comorbidities

                WITH p, t, o, patient_comorbidities, distance,
                     CASE
                       WHEN total_comorbidities > 0
                       THEN toFloat(overlap_count * 2) / total_comorbidities
                       ELSE 0.5
                     END AS comorbidity_similarity

                WITH p, t, o, patient_comorbidities,
                     (1.0 - (distance / 15.0)) * 0.7 + comorbidity_similarity * 0.3 AS similarity_score

                WHERE similarity_score >= 0.5
                ORDER BY similarity_score DESC
                LIMIT $limit

                RETURN p.patient_id AS patient_id,
                       p.age AS age,
                       p.gender AS gender,
                       p.hba1c_baseline AS hba1c,
                       p.c_peptide AS c_peptide,
                       p.bmi AS bmi,
                       p.egfr AS egfr,
                       patient_comorbidities,
                       t.drug_name AS treatment,
                       t.drug_class AS drug_class,
                       o.hba1c_reduction AS reduction,
                       o.hba1c_followup AS hba1c_followup,
                       CASE
                         WHEN o.time_to_target IS NULL OR toString(o.time_to_target) = 'NaN'
                         THEN 'Unknown'
                         ELSE o.time_to_target
                       END AS time_to_target,
                       CASE
                         WHEN o.adverse_events IS NULL OR toString(o.adverse_events) = 'NaN'
                         THEN 'None'
                         ELSE o.adverse_events
                       END AS adverse_events,
                       o.outcome_category AS outcome_category,
                       similarity_score
                """

                result = session.run(
                    query,
                    age_group=age_group,
                    hba1c_severity=hba1c_severity,
                    age=age,
                    hba1c=hba1c,
                    c_peptide=c_peptide,
                    limit=limit,
                    comorbidities=comorbidities,
                    treatment_filter=treatment_filter,
                )

                records = [dict(r) for r in result]

                if not records:
                    return {
                        "nodes": [],
                        "edges": [],
                        "metadata": {
                            "query_patient": patient_profile,
                            "filters_applied": {
                                "age_group": age_group,
                                "hba1c_severity": hba1c_severity,
                                "treatment": treatment_filter,
                                "comorbidities": comorbidities,
                            },
                            "results_found": 0,
                        },
                    }

                return self._build_graph(records, patient_profile, age_group, hba1c_severity, treatment_filter, comorbidities)

        except Exception as e:
            logger.error("Error finding similar cases (graph): %s", e)
            return {"nodes": [], "edges": [], "metadata": {"error": str(e)}}

    # ─── Patient detail by ID ───

    def get_patient_by_id(self, patient_id: str) -> Optional[Dict[str, Any]]:
        if not self.is_connected():
            logger.warning("Neo4j not connected for patient lookup")
            return None

        try:
            with self.driver.session() as session:
                query = """
                MATCH (p:Patient {patient_id: $patient_id})
                OPTIONAL MATCH (p)-[:RECEIVED_TREATMENT]->(t:Treatment)
                OPTIONAL MATCH (o:Outcome {patient_id: $patient_id})
                OPTIONAL MATCH (p)-[:HAS_CONDITION]->(c:Comorbidity)

                WITH p, t, o, collect(DISTINCT c.condition_name) AS comorbidities

                RETURN p.patient_id AS patient_id,
                       p.age AS age,
                       p.gender AS gender,
                       p.ethnicity AS ethnicity,
                       p.hba1c_baseline AS hba1c_baseline,
                       p.diabetes_duration AS diabetes_duration,
                       p.fasting_glucose AS fasting_glucose,
                       p.c_peptide AS c_peptide,
                       p.egfr AS egfr,
                       p.bmi AS bmi,
                       p.bp_systolic AS bp_systolic,
                       p.bp_diastolic AS bp_diastolic,
                       p.alt AS alt,
                       p.ldl AS ldl,
                       p.hdl AS hdl,
                       p.triglycerides AS triglycerides,
                       p.previous_prediabetes AS previous_prediabetes,
                       p.age_group AS age_group,
                       p.bmi_category AS bmi_category,
                       p.hba1c_severity AS hba1c_severity,
                       p.kidney_function AS kidney_function,
                       comorbidities,
                       t.drug_name AS treatment_name,
                       t.drug_class AS drug_class,
                       t.cost_category AS cost_category,
                       t.evidence_level AS evidence_level,
                       o.hba1c_reduction AS hba1c_reduction,
                       o.hba1c_followup AS hba1c_followup,
                       CASE
                         WHEN o.time_to_target IS NULL OR toString(o.time_to_target) = 'NaN'
                         THEN 'Unknown'
                         ELSE o.time_to_target
                       END AS time_to_target,
                       CASE
                         WHEN o.adverse_events IS NULL OR toString(o.adverse_events) = 'NaN'
                         THEN 'None'
                         ELSE o.adverse_events
                       END AS adverse_events,
                       o.outcome_category AS outcome_category,
                       o.success AS success
                """

                record = session.run(query, patient_id=patient_id).single()

                if not record:
                    logger.info("Patient %s not found in Neo4j", patient_id)
                    return None

                patient_data = {
                    "patient_id": record["patient_id"],
                    "demographics": {
                        "age": int(record["age"]),
                        "gender": record["gender"],
                        "ethnicity": record["ethnicity"],
                        "age_group": record["age_group"],
                    },
                    "clinical_features": {
                        "hba1c_baseline": round(float(record["hba1c_baseline"]), 1),
                        "diabetes_duration": round(float(record["diabetes_duration"]), 1),
                        "fasting_glucose": round(float(record["fasting_glucose"]), 1),
                        "c_peptide": round(float(record["c_peptide"]), 2),
                        "egfr": round(float(record["egfr"]), 1),
                        "bmi": round(float(record["bmi"]), 1),
                        "bp_systolic": int(record["bp_systolic"]),
                        "bp_diastolic": int(record["bp_diastolic"]),
                        "alt": round(float(record["alt"]), 1),
                        "ldl": round(float(record["ldl"]), 1),
                        "hdl": round(float(record["hdl"]), 1),
                        "triglycerides": round(float(record["triglycerides"]), 1),
                        "previous_prediabetes": bool(record["previous_prediabetes"]),
                    },
                    "clinical_categories": {
                        "bmi_category": record["bmi_category"],
                        "hba1c_severity": record["hba1c_severity"],
                        "kidney_function": record["kidney_function"],
                    },
                    "comorbidities": record["comorbidities"] or [],
                    "treatment": None,
                    "outcome": None,
                }

                if record["treatment_name"]:
                    patient_data["treatment"] = {
                        "drug_name": record["treatment_name"],
                        "drug_class": record["drug_class"],
                        "cost_category": record["cost_category"],
                        "evidence_level": record["evidence_level"],
                    }

                if record["hba1c_reduction"] is not None:
                    patient_data["outcome"] = {
                        "hba1c_reduction": round(float(record["hba1c_reduction"]), 1),
                        "hba1c_followup": round(float(record["hba1c_followup"]), 1),
                        "time_to_target": record["time_to_target"],
                        "adverse_events": record["adverse_events"],
                        "outcome_category": record["outcome_category"],
                        "success": bool(record["success"]),
                    }

                logger.info("Retrieved patient %s from Neo4j", patient_id)
                return patient_data

        except Exception as e:
            logger.error("Error retrieving patient %s: %s", patient_id, e)
            return None

    # ─── Health check ───

    def health_check(self) -> Dict[str, Any]:
        if not self.is_connected():
            return {"status": "unhealthy", "connected": False}

        try:
            with self.driver.session() as session:
                node_count = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
                rel_count = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]

            return {
                "status": "healthy",
                "connected": True,
                "uri": self.uri,
                "node_count": node_count,
                "relationship_count": rel_count,
            }
        except Exception as e:
            return {"status": "unhealthy", "connected": False, "error": str(e)}

    # ─── Private helpers ───

    def _build_graph(
        self,
        records: List[Dict],
        patient_profile: Dict[str, Any],
        age_group: str,
        hba1c_severity: str,
        treatment_filter: Optional[str],
        comorbidities: List[str],
    ) -> Dict[str, Any]:
        nodes = []
        edges = []

        # Query patient node (centre)
        nodes.append({
            "id": "query_patient",
            "type": "query_patient",
            "label": "Query Patient",
            "data": {
                "age": patient_profile.get("age"),
                "hba1c_baseline": patient_profile.get("hba1c_baseline"),
                "c_peptide": patient_profile.get("c_peptide"),
                "bmi": patient_profile.get("bmi"),
                "comorbidities": comorbidities,
            },
            "style": {"color": "#FF6B6B", "size": "large", "shape": "star"},
        })

        treatment_nodes_added: set[str] = set()

        for idx, rec in enumerate(records):
            pid = rec["patient_id"]
            treatment_name = rec["treatment"]
            similarity = round(rec["similarity_score"], 2)

            # Patient node
            nodes.append({
                "id": pid,
                "type": "patient",
                "label": f"Patient {idx + 1}",
                "data": {
                    "patient_id": pid,
                    "age": int(rec["age"]),
                    "gender": rec["gender"],
                    "hba1c_baseline": round(rec["hba1c"], 1),
                    "c_peptide": round(rec["c_peptide"], 2),
                    "bmi": round(rec["bmi"], 1),
                    "egfr": round(rec["egfr"], 1),
                    "comorbidities": rec["patient_comorbidities"],
                    "similarity_score": similarity,
                },
                "style": {
                    "color": self._similarity_color(similarity),
                    "size": "medium",
                    "shape": "circle",
                },
            })

            # Treatment node
            tid = f"treatment_{treatment_name}"
            if tid not in treatment_nodes_added:
                nodes.append({
                    "id": tid,
                    "type": "treatment",
                    "label": treatment_name,
                    "data": {"treatment": treatment_name, "drug_class": rec["drug_class"]},
                    "style": {"color": "#4ECDC4", "size": "medium", "shape": "square"},
                })
                treatment_nodes_added.add(tid)

            # Outcome node
            oid = f"outcome_{pid}"
            nodes.append({
                "id": oid,
                "type": "outcome",
                "label": f"Outcome {idx + 1}",
                "data": {
                    "hba1c_reduction": round(rec["reduction"], 1),
                    "hba1c_followup": round(rec["hba1c_followup"], 1),
                    "time_to_target": rec["time_to_target"],
                    "adverse_events": rec["adverse_events"],
                    "outcome_category": rec["outcome_category"],
                },
                "style": {
                    "color": self._outcome_color(rec["outcome_category"]),
                    "size": "small",
                    "shape": "diamond",
                },
            })

            # Edges
            edges.append({
                "id": f"edge_query_{pid}",
                "source": "query_patient",
                "target": pid,
                "type": "SIMILAR_TO",
                "label": f"{similarity * 100:.0f}% similar",
                "data": {"similarity_score": similarity},
                "style": {"width": self._edge_width(similarity), "color": "#95A5A6"},
            })
            edges.append({
                "id": f"edge_{pid}_{tid}",
                "source": pid,
                "target": tid,
                "type": "RECEIVED_TREATMENT",
                "label": "received",
                "data": {},
                "style": {"width": 2, "color": "#3498DB"},
            })
            edges.append({
                "id": f"edge_{tid}_{oid}",
                "source": tid,
                "target": oid,
                "type": "RESULTED_IN",
                "label": f"Δ{rec['reduction']:.1f}%",
                "data": {"hba1c_reduction": round(rec["reduction"], 1)},
                "style": {"width": 2, "color": "#2ECC71"},
            })

        metadata = {
            "query_patient": patient_profile,
            "filters_applied": {
                "age_group": age_group,
                "hba1c_severity": hba1c_severity,
                "treatment": treatment_filter,
                "comorbidities": comorbidities,
            },
            "results_found": len(records),
            "similarity_range": {
                "min": round(min(r["similarity_score"] for r in records), 2),
                "max": round(max(r["similarity_score"] for r in records), 2),
                "avg": round(sum(r["similarity_score"] for r in records) / len(records), 2),
            },
        }

        return {"nodes": nodes, "edges": edges, "metadata": metadata}

    @staticmethod
    def _get_age_group(age: float) -> str:
        if age < 40:
            return "<40"
        elif age < 50:
            return "40-50"
        elif age < 60:
            return "50-60"
        elif age < 70:
            return "60-70"
        return ">70"

    @staticmethod
    def _get_hba1c_severity(hba1c: float) -> str:
        if hba1c < 7:
            return "Mild"
        elif hba1c < 8:
            return "Moderate"
        elif hba1c < 9:
            return "Severe"
        return "Very_Severe"

    @staticmethod
    def _similarity_color(score: float) -> str:
        if score >= 0.9:
            return "#27AE60"
        elif score >= 0.8:
            return "#2ECC71"
        elif score >= 0.7:
            return "#F39C12"
        return "#E74C3C"

    @staticmethod
    def _outcome_color(category: str) -> str:
        return {"Success": "#27AE60", "Partial": "#F39C12", "Failure": "#E74C3C"}.get(
            category, "#95A5A6"
        )

    @staticmethod
    def _edge_width(score: float) -> int:
        if score >= 0.9:
            return 5
        elif score >= 0.8:
            return 4
        elif score >= 0.7:
            return 3
        return 2