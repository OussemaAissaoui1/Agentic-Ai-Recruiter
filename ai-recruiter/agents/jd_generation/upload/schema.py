"""Pydantic models for upload validation.

The shape matches the shipped `synthetic_employees.json` so a recruiter can
download that, edit, and re-upload without translation.
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class SkillEntry(BaseModel):
    id:        str
    name:      str
    category:  Optional[str] = None
    esco_id:   Optional[str] = None


class RoleEntry(BaseModel):
    id:           str
    title:        str
    level:        Literal["junior", "mid", "senior"]
    role_family:  str
    department:   Optional[str] = None


class TeamEntry(BaseModel):
    id:          str
    name:        str
    department:  Optional[str] = None
    size:        Optional[int] = None


class EducationEntry(BaseModel):
    id:                str
    degree:            Optional[str] = None
    field:             Optional[str] = None
    institution_tier:  Optional[str] = None


class PriorCompanyEntry(BaseModel):
    id:           str
    name:         str
    industry:     Optional[str] = None
    size_bucket:  Optional[str] = None


class EmployeeRoleHeld(BaseModel):
    role_id:  str
    start:    Optional[str] = None
    end:      Optional[str] = None
    current:  bool = True


class EmployeeSkill(BaseModel):
    skill_id:     str
    proficiency:  int = Field(ge=1, le=5)


class EmployeeEntry(BaseModel):
    id:               str
    hire_date:        Optional[str] = None
    tenure_years:     Optional[float] = None
    level:            Literal["junior", "mid", "senior"]
    status:           str = "active"
    roles_held:       List[EmployeeRoleHeld] = Field(default_factory=list)
    skills:           List[EmployeeSkill] = Field(default_factory=list)
    team_id:          Optional[str] = None
    education:        List[EducationEntry] = Field(default_factory=list)
    prior_companies:  List[PriorCompanyEntry] = Field(default_factory=list)


class PastJDEntry(BaseModel):
    id:            str
    text:          str
    role_family:   str
    status:        Literal["draft", "approved", "rejected"]
    hire_outcome:  Optional[Literal["good_hire", "bad_hire", "not_filled"]] = None
    created_at:    Optional[str] = None


class UploadPayload(BaseModel):
    """The canonical upload shape."""
    skills:     List[SkillEntry] = Field(default_factory=list)
    roles:      List[RoleEntry] = Field(default_factory=list)
    teams:      List[TeamEntry] = Field(default_factory=list)
    employees:  List[EmployeeEntry] = Field(default_factory=list)
    past_jds:   List[PastJDEntry] = Field(default_factory=list)

    def counts(self) -> dict:
        return {
            "employees":  len(self.employees),
            "roles":      len(self.roles),
            "skills":     len(self.skills),
            "teams":      len(self.teams),
            "past_jds":   len(self.past_jds),
        }

    def referential_warnings(self) -> List[str]:
        """Soft warnings — we accept the upload but flag dangling references."""
        warnings: List[str] = []
        skill_ids = {s.id for s in self.skills}
        role_ids = {r.id for r in self.roles}
        team_ids = {t.id for t in self.teams}
        for emp in self.employees:
            for s in emp.skills:
                if s.skill_id not in skill_ids:
                    warnings.append(
                        f"employee {emp.id} references unknown skill_id={s.skill_id}"
                    )
            for r in emp.roles_held:
                if r.role_id not in role_ids:
                    warnings.append(
                        f"employee {emp.id} references unknown role_id={r.role_id}"
                    )
            if emp.team_id and emp.team_id not in team_ids:
                warnings.append(f"employee {emp.id} references unknown team_id={emp.team_id}")
        return warnings[:50]  # cap to avoid runaway responses
