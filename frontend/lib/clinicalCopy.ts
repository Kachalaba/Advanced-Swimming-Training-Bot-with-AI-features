import type { AffectedSide, CaptureQuality } from "./clinical";
import type { RehabLocale } from "./rehabCopy";

type ClinicalCopy = {
  language: string;
  localOnly: string;
  prototype: string;
  backToRehab: string;
  retry: string;
  loading: string;
  workspace: {
    eyebrow: string;
    title: string;
    subtitle: string;
    newPatient: string;
    active: string;
    archived: string;
    noPatientsTitle: string;
    noPatientsBody: string;
    loadError: string;
    latestVisit: string;
    noVisits: string;
    activeEpisode: string;
    noEpisode: string;
    openPatient: string;
  };
  patientForm: {
    title: string;
    athlete: string;
    displayName: string;
    affectedSide: string;
    clinicalContext: string;
    precautions: string;
    create: string;
    cancel: string;
    submitError: string;
  };
  affectedSides: Record<AffectedSide, string>;
  patient: {
    context: string;
    precautions: string;
    noPrecautions: string;
    newVisit: string;
    startEpisode: string;
    progress: string;
    visits: string;
    baseline: string;
    current: string;
    target: string;
    archive: string;
    loadError: string;
  };
  episodeForm: {
    title: string;
    episodeTitle: string;
    protocol: string;
    functionalGoal: string;
    targetLeft: string;
    targetRight: string;
    create: string;
    cancel: string;
    submitError: string;
  };
  visit: {
    eyebrow: string;
    title: string;
    steps: {
      context: string;
      readiness: string;
      analysis: string;
      review: string;
      summary: string;
    };
    source: string;
    live: string;
    upload: string;
    preSessionNote: string;
    continue: string;
    startAnalysis: string;
    reviewVisit: string;
    specialistObservation: string;
    quality: string;
    qualityDetails: string;
    warningAcknowledgement: string;
    finalize: string;
    finalized: string;
    openReport: string;
    patientRecord: string;
    repeatMeasurement: string;
    saveError: string;
  };
  readiness: {
    title: string;
    ready: string;
    warning: string;
    blocked: string;
    waiting: string;
    codes: Record<string, string>;
  };
  quality: Record<CaptureQuality, string>;
  safety: string;
};

const uk: ClinicalCopy = {
  language: "Мова",
  localOnly: "Дані зберігаються лише на цьому комп'ютері",
  prototype: "Дослідницький прототип",
  backToRehab: "До лабораторії руху",
  retry: "Повторити",
  loading: "Завантаження клінічного робочого простору…",
  workspace: {
    eyebrow: "SPRINT AI · Clinical Pilot",
    title: "Кабінет фахівця",
    subtitle:
      "Пацієнти, курси відновлення та контрольні вимірювання в одному локальному робочому просторі.",
    newPatient: "Новий пацієнт",
    active: "Активні",
    archived: "Архів",
    noPatientsTitle: "Ще немає клінічних профілів",
    noPatientsBody:
      "Створіть профіль на основі наявного спортсмена та додайте перший курс відновлення.",
    loadError: "Не вдалося завантажити пацієнтів",
    latestVisit: "Останній візит",
    noVisits: "Візитів ще немає",
    activeEpisode: "Активний курс",
    noEpisode: "Немає активного курсу",
    openPatient: "Відкрити картку",
  },
  patientForm: {
    title: "Створити клінічний профіль",
    athlete: "Пов'язаний спортсмен",
    displayName: "Ім'я для відображення",
    affectedSide: "Сторона обмеження",
    clinicalContext: "Клінічний контекст",
    precautions: "Застереження та протипоказання",
    create: "Створити профіль",
    cancel: "Скасувати",
    submitError: "Не вдалося створити профіль",
  },
  affectedSides: {
    left: "Ліворуч",
    right: "Праворуч",
    bilateral: "Двобічно",
    unspecified: "Не вказано",
  },
  patient: {
    context: "Контекст",
    precautions: "Застереження",
    noPrecautions: "Застереження не вказані",
    newVisit: "Новий візит",
    startEpisode: "Створити курс",
    progress: "Динаміка курсу",
    visits: "Історія візитів",
    baseline: "Базове вимірювання",
    current: "Поточне вимірювання",
    target: "Функціональна ціль",
    archive: "Архівувати",
    loadError: "Не вдалося завантажити картку пацієнта",
  },
  episodeForm: {
    title: "Новий курс реабілітації",
    episodeTitle: "Назва курсу",
    protocol: "Протокол вимірювання",
    functionalGoal: "Функціональна ціль",
    targetLeft: "Цільовий ROM ліворуч",
    targetRight: "Цільовий ROM праворуч",
    create: "Створити курс",
    cancel: "Скасувати",
    submitError: "Не вдалося створити курс",
  },
  visit: {
    eyebrow: "Клінічний візит",
    title: "Нове контрольне вимірювання",
    steps: {
      context: "Контекст",
      readiness: "Готовність",
      analysis: "Аналіз",
      review: "Перевірка",
      summary: "Підсумок",
    },
    source: "Джерело відео",
    live: "Камера наживо",
    upload: "Завантажити відео",
    preSessionNote: "Стан перед сесією",
    continue: "Продовжити",
    startAnalysis: "Почати аналіз",
    reviewVisit: "Перевірити результат",
    specialistObservation: "Спостереження фахівця",
    quality: "Якість вимірювання",
    qualityDetails: "Деталі якості",
    warningAcknowledgement: "Я врахував обмеження цього вимірювання",
    finalize: "Фіналізувати візит",
    finalized: "Візит фіналізовано",
    openReport: "Відкрити звіт",
    patientRecord: "До картки пацієнта",
    repeatMeasurement: "Повторити вимірювання",
    saveError: "Не вдалося зберегти або фіналізувати візит",
  },
  readiness: {
    title: "Готовність до вимірювання",
    ready: "Готово",
    warning: "Потрібне підтвердження",
    blocked: "Аналіз заблоковано",
    waiting: "Очікування сигналу камери",
    codes: {
      ready: "Ключові орієнтири видно, камера готова.",
      pose_missing: "Людину не виявлено в кадрі.",
      target_landmarks_missing: "Цільові суглобові орієнтири не видно.",
      confidence_low: "Впевненість розпізнавання надто низька.",
      confidence_warning: "Впевненість розпізнавання нестабільна.",
      camera_adjust: "Вирівняйте або повторно калібруйте камеру.",
      contralateral_landmarks_missing:
        "Не всі орієнтири протилежної сторони стабільно видно.",
      upload_ready:
        "Відео вибрано. Якість пози буде підтверджено під час аналізу.",
    },
  },
  quality: {
    acceptable: "Прийнятна",
    accepted_with_warning: "Прийнята із застереженням",
    repeat_required: "Потрібне повторне вимірювання",
  },
  safety:
    "SPRINT AI підтримує професійне обговорення руху та не встановлює діагноз, лікування або прогноз.",
};

const en: ClinicalCopy = {
  language: "Language",
  localOnly: "Data is stored only on this computer",
  prototype: "Research prototype",
  backToRehab: "Back to movement lab",
  retry: "Retry",
  loading: "Loading clinical workspace…",
  workspace: {
    eyebrow: "SPRINT AI · Clinical Pilot",
    title: "Specialist workspace",
    subtitle:
      "Patients, rehabilitation episodes, and control measurements in one local workspace.",
    newPatient: "New patient",
    active: "Active",
    archived: "Archived",
    noPatientsTitle: "No clinical profiles yet",
    noPatientsBody:
      "Create a profile from an existing athlete and add the first rehabilitation episode.",
    loadError: "Could not load patients",
    latestVisit: "Latest visit",
    noVisits: "No visits yet",
    activeEpisode: "Active episode",
    noEpisode: "No active episode",
    openPatient: "Open patient",
  },
  patientForm: {
    title: "Create clinical profile",
    athlete: "Linked athlete",
    displayName: "Display name",
    affectedSide: "Affected side",
    clinicalContext: "Clinical context",
    precautions: "Precautions and contraindications",
    create: "Create profile",
    cancel: "Cancel",
    submitError: "Could not create profile",
  },
  affectedSides: {
    left: "Left",
    right: "Right",
    bilateral: "Bilateral",
    unspecified: "Not specified",
  },
  patient: {
    context: "Context",
    precautions: "Precautions",
    noPrecautions: "No precautions recorded",
    newVisit: "New visit",
    startEpisode: "Start episode",
    progress: "Episode progress",
    visits: "Visit history",
    baseline: "Baseline measurement",
    current: "Current measurement",
    target: "Functional goal",
    archive: "Archive",
    loadError: "Could not load patient record",
  },
  episodeForm: {
    title: "New rehabilitation episode",
    episodeTitle: "Episode title",
    protocol: "Measurement protocol",
    functionalGoal: "Functional goal",
    targetLeft: "Target left ROM",
    targetRight: "Target right ROM",
    create: "Create episode",
    cancel: "Cancel",
    submitError: "Could not create episode",
  },
  visit: {
    eyebrow: "Clinical visit",
    title: "New control measurement",
    steps: {
      context: "Context",
      readiness: "Readiness",
      analysis: "Analysis",
      review: "Review",
      summary: "Summary",
    },
    source: "Video source",
    live: "Live camera",
    upload: "Upload video",
    preSessionNote: "Pre-session status",
    continue: "Continue",
    startAnalysis: "Start analysis",
    reviewVisit: "Review result",
    specialistObservation: "Specialist observation",
    quality: "Measurement quality",
    qualityDetails: "Quality details",
    warningAcknowledgement: "I acknowledge this measurement limitation",
    finalize: "Finalize visit",
    finalized: "Visit finalized",
    openReport: "Open report",
    patientRecord: "Patient record",
    repeatMeasurement: "Repeat measurement",
    saveError: "Could not save or finalize the visit",
  },
  readiness: {
    title: "Measurement readiness",
    ready: "Ready",
    warning: "Acknowledgement required",
    blocked: "Analysis blocked",
    waiting: "Waiting for camera signal",
    codes: {
      ready: "Key landmarks are visible and the camera is ready.",
      pose_missing: "No person is detected in the frame.",
      target_landmarks_missing: "Target joint landmarks are not visible.",
      confidence_low: "Pose confidence is too low.",
      confidence_warning: "Pose confidence is unstable.",
      camera_adjust: "Level or recalibrate the camera.",
      contralateral_landmarks_missing:
        "Contralateral landmarks are not consistently visible.",
      upload_ready:
        "Video selected. Pose quality will be confirmed during analysis.",
    },
  },
  quality: {
    acceptable: "Acceptable",
    accepted_with_warning: "Accepted with warning",
    repeat_required: "Repeat required",
  },
  safety:
    "SPRINT AI supports professional movement discussion and does not diagnose, prescribe treatment, or provide prognosis.",
};

export const clinicalCopy: Record<RehabLocale, ClinicalCopy> = { uk, en };
