export type WorkspaceId =
  | "home"
  | "tryon"
  | "plants"
  | "laboratory"
  | "album"
  | "tasks"
  | "models"
  | "settings";

export interface DeviceInfo {
  gpu_name: string | null;
  vram_mib: number | null;
  gpu_available: boolean;
}

export interface DashboardSnapshot {
  onboarding_complete: boolean;
  app_data_dir: string;
  asset_count: number;
  active_task_count: number;
  installed_model_count: number;
  device: DeviceInfo;
}

export interface AssetSummary {
  id: string;
  original_name: string;
  stored_path: string;
  mime_type: string;
  size_bytes: number;
  created_at: string;
}

export interface EngineHealth {
  available: boolean;
  python_version: string | null;
  platform: string | null;
  protocol_version: number | null;
  message: string;
}

export type PreflightLevel = "pass" | "warning" | "block";

export interface PreflightCheck {
  id: string;
  label: string;
  level: PreflightLevel;
  message: string;
  value?: unknown;
}

export interface ImagePreflightResult {
  role: "person" | "garment";
  status: PreflightLevel;
  metrics: {
    width: number;
    height: number;
    megapixels: number;
    file_size_bytes: number;
    format: string;
    brightness?: number;
    edge_score?: number;
  };
  checks: PreflightCheck[];
  limitations: string[];
}

export interface TryOnPreflightResult {
  task_id: string;
  status: PreflightLevel;
  person: ImagePreflightResult;
  garment: ImagePreflightResult;
  scene: {
    supported: boolean;
    blockers: string[];
    person_coverage: number;
    layered_reference: boolean;
    category: "tops" | "bottoms" | "one-pieces";
    category_notice: string;
  };
  ran_at: string;
}

export interface TryOnDraft {
  person: AssetSummary | null;
  garment: AssetSummary | null;
  category: "automatic" | "tops" | "bottoms" | "one-pieces";
  garment_photo_type: "model" | "flat-lay";
  sleeve_mode: "automatic" | "standard" | "reveal-arms";
  updated_at: string | null;
}

export interface TryOnModelStatus {
  installed: boolean;
  ready: boolean;
  model_revision: string;
  source_revision: string;
  human_parser_revision: string;
  python_version: string;
  torch_version: string | null;
  cuda_available: boolean;
  gpu_name: string | null;
  runtime_importable: boolean;
  weights_bytes: number;
  pose_provider: string;
  message: string;
  release_status: string;
}

export interface HandRefinerStatus {
  installed: boolean;
  ready: boolean;
  runtime_installed: boolean;
  source_installed: boolean;
  base_model_installed: boolean;
  control_model_installed: boolean;
  landmark_model_installed: boolean;
  model_revision: string;
  python_version: string;
  required_modules: Record<string, boolean>;
  weights_bytes: number;
  expected_download_bytes: number;
  expected_vram_gib: number;
  automatic_download: false;
  message: string;
  release_status: "evaluation-required" | "single-sample-evaluation-passed" | string;
}

export interface TryOnGenerationResult {
  task_id: string;
  asset: AssetSummary;
  output_path: string;
  width: number;
  height: number;
  elapsed_seconds: number;
  category: TryOnDraft["category"];
  requested_category: TryOnDraft["category"];
  category_notice: string;
  garment_photo_type: TryOnDraft["garment_photo_type"];
  sleeve_mode: TryOnDraft["sleeve_mode"];
  segmentation_free: boolean;
  adaptive_strategy: "preserve-body" | "reconstruct-coverage";
  coverage_notice: string;
  num_timesteps: 20 | 30;
  generation_passes: 1 | 2;
  hand_refinement_used?: boolean;
  hand_refinement_revision?: string;
  skin_tone_delta_before: number;
  skin_tone_delta_after: number;
  verification_status: "passed" | "review";
  quality_checks: {
    garment: { passed: boolean; message: string };
    limbs: {
      complete: boolean;
      expected_arms: number;
      detected_arms: number;
      expected_legs: number;
      detected_legs: number;
      arms: Array<{
        side: "left" | "right";
        segment_coverage: number;
        upper_arm_coverage: number;
        forearm_coverage: number;
        hand_detected: boolean;
        hand_keypoints: number;
        hand_supported_points: number;
        hand_extension_ratio: number;
        hand_detail_score: number;
        hand_palm_ratio: number;
        hand_spread_ratio: number;
        hand_bone_cv: number;
        source_hand_occluded: boolean;
        inferred_joints: string[];
        complete: boolean;
      }>;
      legs: Array<{ side: "left" | "right"; segment_coverage: number; upper_leg_coverage: number; lower_leg_coverage: number; foot_detected: boolean; complete: boolean }>;
      repair_targets: string[];
    };
    skin_tone: { passed: boolean; delta_before: number; delta_after: number; reference_modes: Record<string, "same-position" | "same-side" | "adjacent-forearm" | "global-fallback" | "face-fallback"> };
    phone_preserved: boolean;
  };
  seed: number;
  device: "cuda";
  model_revision: string;
  source_revision: string;
  human_parser_revision: string;
}

export interface TryOnHistoryItem extends TryOnGenerationResult {
  created_at: string;
  person_asset: AssetSummary;
  garment_asset: AssetSummary;
}

export interface TaskSummary {
  id: string;
  task_type: string;
  status: "queued" | "running" | "succeeded" | "failed" | "cancelled";
  stage: string | null;
  created_at: string;
  finished_at: string | null;
  error_code: string | null;
  result_status: PreflightLevel | null;
  cancel_requested: boolean;
}

export interface PhotoSourceSummary {
  id: string;
  path: string;
  label: string;
  recursive: boolean;
  status: string;
  photo_count: number;
  last_scanned_at: string | null;
}

export interface DevicePhotoSummary {
  id: string;
  source_id: string;
  file_name: string;
  absolute_path: string;
  mime_type: string;
  size_bytes: number;
  modified_at: string | null;
  category_id: string | null;
  classification_status: "unclassified" | "classified" | "review";
  confidence: number | null;
  classification_source: "none" | "manual" | "local_model";
  classification_label: string | null;
  thumbnail_path: string | null;
}

export interface CategoryPreference {
  id: string;
  label: string;
  description: string;
  enabled: boolean;
}

export interface AlbumClassifierStatus {
  installed: boolean;
  revision: string;
  size_bytes: number | null;
  engine: string;
  threshold: number;
}

export interface PlantModelStatus {
  installed: boolean;
  ready: boolean;
  model_revision: string;
  catalog_revision: string;
  python_version: string;
  torch_version: string | null;
  runtime_importable: boolean;
  weights_bytes: number;
  supported_taxa: number;
  message: string;
  release_status: string;
}

export interface PlantCandidate {
  taxon_id: string;
  chinese_name: string;
  scientific_name: string;
  score: number;
}

export interface PlantPrediction {
  schema_version: number;
  decision: "candidate" | "uncertain" | "unsupported" | "input_invalid";
  candidates: PlantCandidate[];
  visible_features: string[];
  missing_evidence: string[];
  observation_tags: string[];
  model_revision: string;
  catalog_revision: string;
  device: "cuda" | "cpu";
  width: number;
  height: number;
  elapsed_seconds: number;
  notice: string;
  task_id: string;
  observation_id: string;
}

export interface PlantObservation {
  id: string;
  organ: "whole" | "leaf" | "flower" | "fruit";
  note: string;
  status: "pending" | "confirmed" | "unresolved";
  confirmed_taxon_id: string | null;
  confirmed_chinese_name: string | null;
  prediction: PlantPrediction;
  model_revision: string;
  created_at: string;
  updated_at: string;
  asset: AssetSummary;
}

export interface LabMeasurement {
  object_id: number;
  area_pixels: number;
  area: number;
  equivalent_diameter: number;
  centroid_x: number;
  centroid_y: number;
  touches_edge: boolean;
}

export interface LabResult {
  schema_version: number;
  algorithm_revision: string;
  object_type: "nucleus";
  review_status: "unreviewed";
  metadata: { width: number; height: number; channels: number; source_mode: string; source_dtype: string };
  parameters: { threshold_scale: number; otsu_threshold: number; min_area: number; exclude_edges: boolean; scale_um_per_px: number | null };
  summary: { count: number; unit: "px" | "μm"; area_unit: "px²" | "μm²"; mean_area: number | null; median_area: number | null; mean_equivalent_diameter: number | null };
  measurements: LabMeasurement[];
  overlay_path: string;
  elapsed_seconds: number;
  limitations: string[];
  task_id: string;
  analysis_id: string;
}

export interface LabAnalysis {
  id: string;
  result: LabResult;
  overlay_path: string;
  review_status: "unreviewed" | "reviewed";
  created_at: string;
  updated_at: string;
  asset: AssetSummary;
}

export interface CloudSession {
  access_token: string;
  access_expires_in: number;
  refresh_token: string;
  refresh_expires_in: number;
  device_id: string;
  user_id: string;
  display_name: string;
}
