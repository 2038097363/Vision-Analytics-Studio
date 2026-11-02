import { convertFileSrc, invoke } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";
import type {
  AssetSummary,
  AlbumClassifierStatus,
  DashboardSnapshot,
  EngineHealth,
  HandRefinerStatus,
  CategoryPreference,
  DevicePhotoSummary,
  PhotoSourceSummary,
  PlantModelStatus,
  PlantObservation,
  LabAnalysis,
  TaskSummary,
  TryOnDraft,
  TryOnGenerationResult,
  TryOnHistoryItem,
  TryOnModelStatus,
  TryOnPreflightResult,
} from "../types";

const inDesktop = () => "__TAURI_INTERNALS__" in window;

const browserSnapshot: DashboardSnapshot = {
  onboarding_complete: false,
  app_data_dir: "浏览器预览不会写入本地数据",
  asset_count: 0,
  active_task_count: 0,
  installed_model_count: 0,
  device: { gpu_name: null, vram_mib: null, gpu_available: false },
};

export async function initializeApp(): Promise<DashboardSnapshot> {
  if (!inDesktop()) return browserSnapshot;
  return invoke<DashboardSnapshot>("initialize_app");
}

export async function completeOnboarding(): Promise<DashboardSnapshot> {
  if (!inDesktop()) return { ...browserSnapshot, onboarding_complete: true };
  return invoke<DashboardSnapshot>("complete_onboarding");
}

export async function chooseAndImportAssets(): Promise<AssetSummary[]> {
  if (!inDesktop()) throw new Error("请使用桌面模式导入本地图片");
  const selected = await open({
    multiple: true,
    directory: false,
    filters: [{ name: "图片", extensions: ["jpg", "jpeg", "png", "webp", "tif", "tiff"] }],
  });
  if (!selected) return [];
  const paths = Array.isArray(selected) ? selected : [selected];
  return Promise.all(paths.map((path) => invoke<AssetSummary>("import_asset", { path })));
}

export async function chooseAndImportAsset(): Promise<AssetSummary | null> {
  if (!inDesktop()) throw new Error("请使用桌面模式导入本地图片");
  const selected = await open({
    multiple: false,
    directory: false,
    filters: [{ name: "图片", extensions: ["jpg", "jpeg", "png", "webp", "tif", "tiff"] }],
  });
  if (!selected || Array.isArray(selected)) return null;
  return invoke<AssetSummary>("import_asset", { path: selected });
}

export function assetUrl(asset: AssetSummary): string {
  return inDesktop() ? convertFileSrc(asset.stored_path) : "";
}

export async function listRecentAssets(): Promise<AssetSummary[]> {
  if (!inDesktop()) return [];
  return invoke<AssetSummary[]>("list_recent_assets", { limit: 6 });
}

export async function choosePhotoSource(): Promise<PhotoSourceSummary | null> {
  if (!inDesktop()) throw new Error("请使用桌面模式授权照片文件夹");
  const selected = await open({ multiple: false, directory: true });
  if (!selected || Array.isArray(selected)) return null;
  return invoke<PhotoSourceSummary>("add_photo_source", { path: selected });
}

export async function listPhotoSources(): Promise<PhotoSourceSummary[]> {
  if (!inDesktop()) return [];
  return invoke<PhotoSourceSummary[]>("list_photo_sources");
}

export async function removePhotoSource(sourceId: string): Promise<void> {
  if (!inDesktop()) return;
  return invoke<void>("remove_photo_source", { sourceId });
}

export async function startPhotoScan(sourceId: string): Promise<string> {
  if (!inDesktop()) throw new Error("请使用桌面模式扫描照片文件夹");
  return invoke<string>("start_photo_scan", { sourceId });
}

export async function cancelPhotoScan(sourceId: string): Promise<void> {
  if (!inDesktop()) return;
  return invoke<void>("cancel_photo_scan", { sourceId });
}

export async function listUncategorizedPhotos(): Promise<DevicePhotoSummary[]> {
  if (!inDesktop()) return [];
  return invoke<DevicePhotoSummary[]>("list_uncategorized_photos", { limit: 200 });
}

export async function listClassifiedPhotos(): Promise<DevicePhotoSummary[]> {
  if (!inDesktop()) return [];
  return invoke<DevicePhotoSummary[]>("list_classified_photos", { limit: 500 });
}

export function devicePhotoThumbnailUrl(photo: DevicePhotoSummary): string {
  return inDesktop() && photo.thumbnail_path ? convertFileSrc(photo.thumbnail_path) : "";
}

export async function setPhotoCategory(photoId: string, categoryId: string | null): Promise<void> {
  if (!inDesktop()) return;
  return invoke<void>("set_photo_category", { photoId, categoryId });
}

export async function setPhotoCategories(photoIds: string[], categoryId: string): Promise<number> {
  if (!inDesktop()) return 0;
  return invoke<number>("set_photo_categories", { photoIds, categoryId });
}

export async function importDevicePhoto(photoId: string): Promise<AssetSummary> {
  if (!inDesktop()) throw new Error("请使用桌面模式导入设备照片");
  return invoke<AssetSummary>("import_device_photo", { photoId });
}

export async function getCategoryPreferences(): Promise<CategoryPreference[]> {
  if (!inDesktop()) return [];
  return invoke<CategoryPreference[]>("get_category_preferences");
}

export async function saveCategoryPreferences(enabledIds: string[]): Promise<CategoryPreference[]> {
  if (!inDesktop()) return [];
  return invoke<CategoryPreference[]>("save_category_preferences", { enabledIds });
}

export async function getAlbumClassifierStatus(): Promise<AlbumClassifierStatus> {
  if (!inDesktop()) return {
    installed: false,
    revision: "mobilenet-v3-small-imagenet1k-v1",
    size_bytes: null,
    engine: "PyTorch · MobileNetV3 Small",
    threshold: 0.35,
  };
  return invoke<AlbumClassifierStatus>("get_album_classifier_status");
}

export async function startAlbumClassification(): Promise<string> {
  if (!inDesktop()) throw new Error("请使用桌面模式运行本地自动分类");
  return invoke<string>("start_album_classification");
}

export async function listTasks(): Promise<TaskSummary[]> {
  if (!inDesktop()) return [];
  return invoke<TaskSummary[]>("list_tasks", { limit: 30 });
}

export async function cancelTask(taskId: string): Promise<void> {
  if (!inDesktop()) return;
  return invoke<void>("cancel_task", { taskId });
}

export async function checkEngine(): Promise<EngineHealth> {
  if (!inDesktop()) {
    return {
      available: false,
      python_version: null,
      platform: null,
      protocol_version: null,
      message: "浏览器预览模式未连接本地引擎",
    };
  }
  return invoke<EngineHealth>("check_engine_health");
}

export async function loadTryOnDraft(): Promise<TryOnDraft> {
  if (!inDesktop()) return { person: null, garment: null, category: "automatic", garment_photo_type: "model", sleeve_mode: "automatic", updated_at: null };
  return invoke<TryOnDraft>("load_tryon_draft");
}

export async function saveTryOnDraft(
  personAssetId: string | null,
  garmentAssetId: string | null,
  category: TryOnDraft["category"] = "automatic",
  garmentPhotoType: TryOnDraft["garment_photo_type"] = "model",
  sleeveMode: TryOnDraft["sleeve_mode"] = "automatic",
): Promise<TryOnDraft> {
  if (!inDesktop()) return { person: null, garment: null, category, garment_photo_type: garmentPhotoType, sleeve_mode: sleeveMode, updated_at: null };
  return invoke<TryOnDraft>("save_tryon_draft", { personAssetId, garmentAssetId, category, garmentPhotoType, sleeveMode });
}

export async function runTryOnPreflight(
  personAssetId: string,
  garmentAssetId: string,
  garmentPhotoType: "model" | "flat-lay",
): Promise<TryOnPreflightResult> {
  if (!inDesktop()) throw new Error("请使用桌面模式运行本地图片预检");
  return invoke<TryOnPreflightResult>("run_tryon_preflight", { personAssetId, garmentAssetId, garmentPhotoType });
}

export async function getTryOnModelStatus(): Promise<TryOnModelStatus> {
  if (!inDesktop()) {
    return {
      installed: false,
      ready: false,
      model_revision: "",
      source_revision: "",
      human_parser_revision: "",
      python_version: "",
      torch_version: null,
      cuda_available: false,
      gpu_name: null,
      runtime_importable: false,
      weights_bytes: 0,
      pose_provider: "",
      message: "浏览器预览模式未连接试衣运行时",
      release_status: "unavailable",
    };
  }
  return invoke<TryOnModelStatus>("get_tryon_model_status");
}

export async function getHandRefinerStatus(): Promise<HandRefinerStatus> {
  if (!inDesktop()) {
    return {
      installed: false,
      ready: false,
      runtime_installed: false,
      source_installed: false,
      base_model_installed: false,
      control_model_installed: false,
      landmark_model_installed: false,
      model_revision: "sd15-hand-inpaint-local-v1",
      python_version: "",
      required_modules: {},
      weights_bytes: 0,
      expected_download_bytes: 2_140_000_000,
      expected_vram_gib: 8,
      automatic_download: false,
      message: "浏览器预览模式未连接手部修复运行时",
      release_status: "evaluation-required",
    };
  }
  return invoke<HandRefinerStatus>("get_hand_refiner_status");
}

export async function runTryOnGeneration(
  personAssetId: string,
  garmentAssetId: string,
  category: TryOnDraft["category"],
  garmentPhotoType: TryOnDraft["garment_photo_type"],
  sleeveMode: TryOnDraft["sleeve_mode"],
  numTimesteps: 20 | 30,
): Promise<TryOnGenerationResult> {
  if (!inDesktop()) throw new Error("请使用桌面模式运行本地虚拟穿搭");
  return invoke<TryOnGenerationResult>("run_tryon_generation", {
    personAssetId,
    garmentAssetId,
    category,
    garmentPhotoType,
    sleeveMode,
    numTimesteps,
  });
}

export function tryOnResultUrl(result: TryOnGenerationResult): string {
  return inDesktop() ? convertFileSrc(result.output_path) : "";
}

export async function listTryOnResults(): Promise<TryOnHistoryItem[]> {
  if (!inDesktop()) return [];
  return invoke<TryOnHistoryItem[]>("list_tryon_results", { limit: 20 });
}

export async function getPlantModelStatus(): Promise<PlantModelStatus> {
  if (!inDesktop()) return {
    installed: false,
    ready: false,
    model_revision: "",
    catalog_revision: "",
    python_version: "",
    torch_version: null,
    runtime_importable: false,
    weights_bytes: 0,
    supported_taxa: 20,
    message: "浏览器预览模式未连接植物识别运行时",
    release_status: "unavailable",
  };
  return invoke<PlantModelStatus>("get_plant_model_status");
}

export async function identifyPlant(
  assetId: string,
  organ: PlantObservation["organ"],
  note: string,
): Promise<PlantObservation> {
  if (!inDesktop()) throw new Error("请使用桌面模式运行本地植物识别");
  return invoke<PlantObservation>("identify_plant", { assetId, organ, note });
}

export async function listPlantObservations(): Promise<PlantObservation[]> {
  if (!inDesktop()) return [];
  return invoke<PlantObservation[]>("list_plant_observations", { limit: 40 });
}

export async function confirmPlantObservation(
  observationId: string,
  taxonId: string | null,
): Promise<PlantObservation> {
  if (!inDesktop()) throw new Error("请使用桌面模式保存植物确认结果");
  return invoke<PlantObservation>("confirm_plant_observation", { observationId, taxonId });
}

export async function runLabSegmentation(
  assetId: string,
  thresholdScale: number,
  minArea: number,
  excludeEdges: boolean,
  scaleUmPerPx: number | null,
): Promise<LabAnalysis> {
  if (!inDesktop()) throw new Error("请使用桌面模式运行实验室图像分析");
  return invoke<LabAnalysis>("run_lab_segmentation", { assetId, thresholdScale, minArea, excludeEdges, scaleUmPerPx });
}

export async function listLabAnalyses(): Promise<LabAnalysis[]> {
  if (!inDesktop()) return [];
  return invoke<LabAnalysis[]>("list_lab_analyses", { limit: 30 });
}

export async function reviewLabAnalysis(analysisId: string): Promise<LabAnalysis> {
  if (!inDesktop()) throw new Error("请使用桌面模式保存复核状态");
  return invoke<LabAnalysis>("review_lab_analysis", { analysisId });
}

export function labOverlayUrl(analysis: LabAnalysis): string {
  return inDesktop() ? convertFileSrc(analysis.overlay_path) : "";
}
