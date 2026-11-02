use chrono::Utc;
use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File},
    io::{BufWriter, Read, Write},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    time::{Duration, SystemTime},
};
use tauri::Manager;
use uuid::Uuid;

#[cfg(windows)]
use std::os::windows::fs::MetadataExt;

#[derive(Serialize)]
struct DeviceInfo {
    gpu_name: Option<String>,
    vram_mib: Option<u64>,
    gpu_available: bool,
}

#[derive(Serialize)]
struct DashboardSnapshot {
    onboarding_complete: bool,
    app_data_dir: String,
    asset_count: u64,
    active_task_count: u64,
    installed_model_count: u64,
    device: DeviceInfo,
}

#[derive(Serialize)]
struct AssetSummary {
    id: String,
    original_name: String,
    stored_path: String,
    mime_type: String,
    size_bytes: u64,
    created_at: String,
}

#[derive(Serialize)]
struct TryOnDraft {
    person: Option<AssetSummary>,
    garment: Option<AssetSummary>,
    category: String,
    garment_photo_type: String,
    sleeve_mode: String,
    updated_at: Option<String>,
}

#[derive(Serialize)]
struct TaskSummary {
    id: String,
    task_type: String,
    status: String,
    stage: Option<String>,
    created_at: String,
    finished_at: Option<String>,
    error_code: Option<String>,
    result_status: Option<String>,
    cancel_requested: bool,
}

#[derive(Serialize)]
struct PhotoSourceSummary {
    id: String,
    path: String,
    label: String,
    recursive: bool,
    status: String,
    photo_count: u64,
    last_scanned_at: Option<String>,
}

#[derive(Serialize)]
struct PhotoIndexSummary {
    id: String,
    source_id: String,
    file_name: String,
    absolute_path: String,
    mime_type: String,
    size_bytes: u64,
    modified_at: Option<String>,
    category_id: Option<String>,
    classification_status: String,
    confidence: Option<f64>,
    classification_source: String,
    classification_label: Option<String>,
    thumbnail_path: Option<String>,
}

#[derive(Serialize)]
struct CategoryPreference {
    id: String,
    label: String,
    description: String,
    enabled: bool,
}

#[derive(Serialize)]
struct AlbumClassifierStatus {
    installed: bool,
    revision: String,
    size_bytes: Option<u64>,
    engine: String,
    threshold: f64,
}

#[derive(Serialize)]
struct EngineHealth {
    available: bool,
    python_version: Option<String>,
    platform: Option<String>,
    protocol_version: Option<u64>,
    message: String,
}

#[derive(Deserialize)]
struct EngineEnvelope {
    protocol_version: u64,
    success: bool,
    payload: Option<serde_json::Value>,
    error: Option<EngineError>,
}

#[derive(Deserialize)]
struct EngineError {
    code: String,
    message: String,
}

fn app_data_dir(app: &tauri::AppHandle) -> Result<PathBuf, String> {
    let dir = app
        .path()
        .app_data_dir()
        .map_err(|error| error.to_string())?;
    fs::create_dir_all(&dir).map_err(|error| error.to_string())?;
    fs::create_dir_all(dir.join("assets")).map_err(|error| error.to_string())?;
    fs::create_dir_all(dir.join("tmp")).map_err(|error| error.to_string())?;
    fs::create_dir_all(dir.join("thumbnails")).map_err(|error| error.to_string())?;
    Ok(dir)
}

fn open_database(app: &tauri::AppHandle) -> Result<Connection, String> {
    let path = app_data_dir(app)?.join("vision-analytics.db");
    let connection = Connection::open(path).map_err(|error| error.to_string())?;
    connection
        .busy_timeout(Duration::from_secs(5))
        .map_err(|error| error.to_string())?;
    connection
        .execute_batch(
            "PRAGMA journal_mode = WAL;
             PRAGMA foreign_keys = ON;
             CREATE TABLE IF NOT EXISTS app_settings (
               key TEXT PRIMARY KEY NOT NULL,
               value TEXT NOT NULL,
               updated_at TEXT NOT NULL
             );
             CREATE TABLE IF NOT EXISTS assets (
               id TEXT PRIMARY KEY NOT NULL,
               original_name TEXT NOT NULL,
               source_path TEXT NOT NULL,
               stored_path TEXT NOT NULL,
               mime_type TEXT NOT NULL,
               size_bytes INTEGER NOT NULL,
               sha256 TEXT NOT NULL,
               created_at TEXT NOT NULL,
               deleted_at TEXT
             );
             CREATE INDEX IF NOT EXISTS idx_assets_created_at ON assets(created_at DESC);
             CREATE INDEX IF NOT EXISTS idx_assets_sha256 ON assets(sha256);
             CREATE TABLE IF NOT EXISTS tasks (
               id TEXT PRIMARY KEY NOT NULL,
               task_type TEXT NOT NULL,
               status TEXT NOT NULL CHECK(status IN ('queued','running','succeeded','failed','cancelled')),
               stage TEXT,
               parameters_json TEXT NOT NULL DEFAULT '{}',
               created_at TEXT NOT NULL,
               started_at TEXT,
               finished_at TEXT,
               error_code TEXT,
               result_json TEXT,
               cancel_requested INTEGER NOT NULL DEFAULT 0
             );
             CREATE TABLE IF NOT EXISTS installed_models (
               id TEXT PRIMARY KEY NOT NULL,
               revision TEXT NOT NULL,
               status TEXT NOT NULL,
               installed_at TEXT NOT NULL
             );
             CREATE TABLE IF NOT EXISTS asset_analysis (
               asset_id TEXT NOT NULL,
               analysis_type TEXT NOT NULL,
               revision TEXT NOT NULL,
               status TEXT NOT NULL,
               result_json TEXT NOT NULL,
               created_at TEXT NOT NULL,
               PRIMARY KEY(asset_id, analysis_type, revision),
               FOREIGN KEY(asset_id) REFERENCES assets(id)
             );
             CREATE TABLE IF NOT EXISTS tryon_drafts (
               id TEXT PRIMARY KEY NOT NULL CHECK(id = 'current'),
               person_asset_id TEXT,
               garment_asset_id TEXT,
               category TEXT NOT NULL DEFAULT 'automatic',
               garment_photo_type TEXT NOT NULL DEFAULT 'model',
               sleeve_mode TEXT NOT NULL DEFAULT 'automatic',
               updated_at TEXT NOT NULL,
               FOREIGN KEY(person_asset_id) REFERENCES assets(id),
               FOREIGN KEY(garment_asset_id) REFERENCES assets(id)
             );
             CREATE TABLE IF NOT EXISTS photo_sources (
               id TEXT PRIMARY KEY NOT NULL,
               path TEXT NOT NULL UNIQUE,
               label TEXT NOT NULL,
               recursive INTEGER NOT NULL DEFAULT 1,
               status TEXT NOT NULL DEFAULT 'ready',
               photo_count INTEGER NOT NULL DEFAULT 0,
               created_at TEXT NOT NULL,
               last_scanned_at TEXT
             );
             CREATE TABLE IF NOT EXISTS device_photos (
               id TEXT PRIMARY KEY NOT NULL,
               source_id TEXT NOT NULL,
               absolute_path TEXT NOT NULL,
               file_name TEXT NOT NULL,
               mime_type TEXT NOT NULL,
               size_bytes INTEGER NOT NULL,
               modified_at TEXT,
               category_id TEXT,
               classification_status TEXT NOT NULL DEFAULT 'unclassified',
               confidence REAL,
               classification_source TEXT NOT NULL DEFAULT 'none',
               classification_revision TEXT,
               classification_label TEXT,
               thumbnail_path TEXT,
               thumbnail_status TEXT NOT NULL DEFAULT 'pending',
               is_present INTEGER NOT NULL DEFAULT 1,
               scan_marker TEXT,
               created_at TEXT NOT NULL,
               updated_at TEXT NOT NULL,
               UNIQUE(source_id, absolute_path),
               FOREIGN KEY(source_id) REFERENCES photo_sources(id) ON DELETE CASCADE
             );
             CREATE INDEX IF NOT EXISTS idx_device_photos_source ON device_photos(source_id, is_present);
             CREATE INDEX IF NOT EXISTS idx_device_photos_category ON device_photos(category_id, classification_status, is_present);
             CREATE TABLE IF NOT EXISTS category_preferences (
               category_id TEXT PRIMARY KEY NOT NULL,
               enabled INTEGER NOT NULL,
               updated_at TEXT NOT NULL
             );
             CREATE TABLE IF NOT EXISTS plant_observations (
               id TEXT PRIMARY KEY NOT NULL,
               asset_id TEXT NOT NULL,
               organ TEXT NOT NULL CHECK(organ IN ('whole','leaf','flower','fruit')),
               note TEXT NOT NULL DEFAULT '',
               status TEXT NOT NULL CHECK(status IN ('pending','confirmed','unresolved')),
               confirmed_taxon_id TEXT,
               confirmed_chinese_name TEXT,
               prediction_json TEXT NOT NULL,
               model_revision TEXT NOT NULL,
               created_at TEXT NOT NULL,
               updated_at TEXT NOT NULL,
               FOREIGN KEY(asset_id) REFERENCES assets(id)
             );
             CREATE TABLE IF NOT EXISTS lab_analyses (
               id TEXT PRIMARY KEY NOT NULL,
               asset_id TEXT NOT NULL,
               result_json TEXT NOT NULL,
               overlay_path TEXT NOT NULL,
               review_status TEXT NOT NULL CHECK(review_status IN ('unreviewed','reviewed')),
               created_at TEXT NOT NULL,
               updated_at TEXT NOT NULL,
               FOREIGN KEY(asset_id) REFERENCES assets(id)
             );",
        )
        .map_err(|error| error.to_string())?;

    let has_task_result = {
        let mut statement = connection
            .prepare("PRAGMA table_info(tasks)")
            .map_err(|error| error.to_string())?;
        let columns = statement
            .query_map([], |row| row.get::<_, String>(1))
            .map_err(|error| error.to_string())?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| error.to_string())?;
        columns.iter().any(|column| column == "result_json")
    };
    if !has_task_result {
        connection
            .execute("ALTER TABLE tasks ADD COLUMN result_json TEXT", [])
            .map_err(|error| error.to_string())?;
    }
    let task_columns = {
        let mut statement = connection
            .prepare("PRAGMA table_info(tasks)")
            .map_err(|error| error.to_string())?;
        let columns = statement
            .query_map([], |row| row.get::<_, String>(1))
            .map_err(|error| error.to_string())?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| error.to_string())?;
        columns
    };
    if !task_columns
        .iter()
        .any(|column| column == "cancel_requested")
    {
        connection
            .execute(
                "ALTER TABLE tasks ADD COLUMN cancel_requested INTEGER NOT NULL DEFAULT 0",
                [],
            )
            .map_err(|error| error.to_string())?;
    }
    let device_photo_columns = {
        let mut statement = connection
            .prepare("PRAGMA table_info(device_photos)")
            .map_err(|error| error.to_string())?;
        let columns = statement
            .query_map([], |row| row.get::<_, String>(1))
            .map_err(|error| error.to_string())?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| error.to_string())?;
        columns
    };
    if !device_photo_columns
        .iter()
        .any(|column| column == "thumbnail_path")
    {
        connection
            .execute(
                "ALTER TABLE device_photos ADD COLUMN thumbnail_path TEXT",
                [],
            )
            .map_err(|error| error.to_string())?;
    }
    let tryon_draft_columns = {
        let mut statement = connection
            .prepare("PRAGMA table_info(tryon_drafts)")
            .map_err(|error| error.to_string())?;
        let columns = statement
            .query_map([], |row| row.get::<_, String>(1))
            .map_err(|error| error.to_string())?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| error.to_string())?;
        columns
    };
    if !tryon_draft_columns
        .iter()
        .any(|column| column == "garment_photo_type")
    {
        connection
            .execute(
                "ALTER TABLE tryon_drafts ADD COLUMN garment_photo_type TEXT NOT NULL DEFAULT 'model'",
                [],
            )
            .map_err(|error| error.to_string())?;
    }
    if !tryon_draft_columns
        .iter()
        .any(|column| column == "sleeve_mode")
    {
        connection
            .execute(
                "ALTER TABLE tryon_drafts ADD COLUMN sleeve_mode TEXT NOT NULL DEFAULT 'automatic'",
                [],
            )
            .map_err(|error| error.to_string())?;
    }
    connection
        .execute(
            "UPDATE tryon_drafts SET category = 'tops' WHERE category = 'upper'",
            [],
        )
        .map_err(|error| error.to_string())?;
    if !device_photo_columns
        .iter()
        .any(|column| column == "thumbnail_status")
    {
        connection
            .execute(
                "ALTER TABLE device_photos ADD COLUMN thumbnail_status TEXT NOT NULL DEFAULT 'pending'",
                [],
            )
            .map_err(|error| error.to_string())?;
    }
    if !device_photo_columns
        .iter()
        .any(|column| column == "classification_source")
    {
        connection
            .execute(
                "ALTER TABLE device_photos ADD COLUMN classification_source TEXT NOT NULL DEFAULT 'none'",
                [],
            )
            .map_err(|error| error.to_string())?;
    }
    if !device_photo_columns
        .iter()
        .any(|column| column == "classification_revision")
    {
        connection
            .execute(
                "ALTER TABLE device_photos ADD COLUMN classification_revision TEXT",
                [],
            )
            .map_err(|error| error.to_string())?;
    }
    if !device_photo_columns
        .iter()
        .any(|column| column == "classification_label")
    {
        connection
            .execute(
                "ALTER TABLE device_photos ADD COLUMN classification_label TEXT",
                [],
            )
            .map_err(|error| error.to_string())?;
    }
    let now = Utc::now().to_rfc3339();
    for category_id in safe_category_ids() {
        connection
            .execute(
                "INSERT OR IGNORE INTO category_preferences(category_id, enabled, updated_at) VALUES(?1, 0, ?2)",
                params![category_id, now],
            )
            .map_err(|error| error.to_string())?;
    }
    Ok(connection)
}

fn safe_categories() -> &'static [(&'static str, &'static str, &'static str)] {
    &[
        ("nature", "风景与自然", "山水、天空、海边与自然景观"),
        ("plants", "植物与花卉", "花朵、树木与非敏感植物画面"),
        ("animals", "动物与宠物", "宠物、鸟类与常见动物"),
        ("food", "美食", "食物、饮品与餐桌画面"),
        ("architecture", "建筑与城市", "建筑外观、街景与公共空间"),
        ("outfits", "服装与穿搭", "衣物、鞋包与穿搭参考"),
        ("art", "艺术与插画", "绘画、手作、展品与插画"),
        ("transport", "交通工具", "车辆、船只与公共交通"),
    ]
}

fn safe_category_ids() -> impl Iterator<Item = &'static str> {
    safe_categories().iter().map(|(id, _, _)| *id)
}

const ALBUM_CLASSIFIER_REVISION: &str = "mobilenet-v3-small-imagenet1k-v1";
const ALBUM_CLASSIFIER_FILE: &str = "mobilenet_v3_small-047dcff4.pth";

fn album_classifier_path(app: &tauri::AppHandle) -> Result<PathBuf, String> {
    let installed = app_data_dir(app)?
        .join("models")
        .join("album")
        .join(ALBUM_CLASSIFIER_FILE);
    if installed.is_file() {
        return Ok(installed);
    }
    Ok(project_root()
        .join("models")
        .join("album")
        .join(ALBUM_CLASSIFIER_FILE))
}

#[tauri::command]
fn get_album_classifier_status(app: tauri::AppHandle) -> Result<AlbumClassifierStatus, String> {
    let path = album_classifier_path(&app)?;
    let metadata = path.metadata().ok();
    Ok(AlbumClassifierStatus {
        installed: metadata.is_some(),
        revision: ALBUM_CLASSIFIER_REVISION.to_string(),
        size_bytes: metadata.map(|value| value.len()),
        engine: "PyTorch · MobileNetV3 Small".to_string(),
        threshold: 0.35,
    })
}

fn scalar_count(connection: &Connection, sql: &str) -> Result<u64, String> {
    connection
        .query_row(sql, [], |row| row.get::<_, u64>(0))
        .map_err(|error| error.to_string())
}

fn detect_device() -> DeviceInfo {
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,memory.total",
            "--format=csv,noheader,nounits",
        ])
        .output();

    if let Ok(output) = output {
        if output.status.success() {
            let line = String::from_utf8_lossy(&output.stdout)
                .lines()
                .next()
                .unwrap_or("")
                .trim()
                .to_string();
            let parts: Vec<_> = line.rsplitn(2, ',').map(str::trim).collect();
            if parts.len() == 2 {
                return DeviceInfo {
                    gpu_name: Some(parts[1].to_string()),
                    vram_mib: parts[0].parse::<u64>().ok(),
                    gpu_available: true,
                };
            }
        }
    }

    DeviceInfo {
        gpu_name: None,
        vram_mib: None,
        gpu_available: false,
    }
}

fn dashboard_snapshot(app: &tauri::AppHandle) -> Result<DashboardSnapshot, String> {
    let data_dir = app_data_dir(app)?;
    let connection = open_database(app)?;
    let onboarding_complete = connection
        .query_row(
            "SELECT value FROM app_settings WHERE key = 'onboarding_complete'",
            [],
            |row| row.get::<_, String>(0),
        )
        .map(|value| value == "true")
        .unwrap_or(false);

    Ok(DashboardSnapshot {
        onboarding_complete,
        app_data_dir: data_dir.to_string_lossy().to_string(),
        asset_count: scalar_count(
            &connection,
            "SELECT COUNT(*) FROM assets WHERE deleted_at IS NULL",
        )?,
        active_task_count: scalar_count(
            &connection,
            "SELECT COUNT(*) FROM tasks WHERE status IN ('queued','running')",
        )?,
        installed_model_count: scalar_count(
            &connection,
            "SELECT COUNT(*) FROM installed_models WHERE status = 'available'",
        )?,
        device: detect_device(),
    })
}

#[tauri::command]
fn initialize_app(app: tauri::AppHandle) -> Result<DashboardSnapshot, String> {
    dashboard_snapshot(&app)
}

fn recover_interrupted_tasks(app: &tauri::AppHandle) -> Result<(), String> {
    let connection = open_database(&app)?;
    let now = Utc::now().to_rfc3339();
    connection
        .execute(
            "UPDATE tasks SET
               status = CASE WHEN cancel_requested = 1 THEN 'cancelled' ELSE 'failed' END,
               finished_at = ?1,
               error_code = CASE WHEN cancel_requested = 1 THEN NULL ELSE 'interrupted_by_restart' END
             WHERE task_type IN ('album.scan','album.classify','tryon.generate','plant.identify','lab.segment') AND status IN ('queued','running')",
            [&now],
        )
        .map_err(|error| error.to_string())?;
    connection
        .execute(
            "UPDATE photo_sources SET status = 'interrupted'
             WHERE status IN ('queued','scanning','thumbnailing')",
            [],
        )
        .map_err(|error| error.to_string())?;
    drop(connection);
    Ok(())
}

#[tauri::command]
fn complete_onboarding(app: tauri::AppHandle) -> Result<DashboardSnapshot, String> {
    let connection = open_database(&app)?;
    connection
        .execute(
            "INSERT INTO app_settings(key, value, updated_at) VALUES('onboarding_complete', 'true', ?1)
             ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
            [Utc::now().to_rfc3339()],
        )
        .map_err(|error| error.to_string())?;
    dashboard_snapshot(&app)
}

fn sha256_file(path: &Path) -> Result<String, String> {
    let mut file = File::open(path).map_err(|error| error.to_string())?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = file.read(&mut buffer).map_err(|error| error.to_string())?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn allowed_extension(path: &Path) -> bool {
    matches!(
        path.extension()
            .and_then(|value| value.to_str())
            .map(str::to_ascii_lowercase)
            .as_deref(),
        Some("jpg" | "jpeg" | "png" | "webp" | "tif" | "tiff")
    )
}

#[tauri::command]
fn import_asset(app: tauri::AppHandle, path: String) -> Result<AssetSummary, String> {
    let source = PathBuf::from(path)
        .canonicalize()
        .map_err(|_| "无法读取所选文件".to_string())?;
    if !source.is_file() || !allowed_extension(&source) {
        return Err("当前只支持 JPEG、PNG、WebP 和二维 TIFF 文件".to_string());
    }
    let metadata = source.metadata().map_err(|error| error.to_string())?;
    if metadata.len() > 100 * 1024 * 1024 {
        return Err("文件超过当前 100MB 导入上限".to_string());
    }

    let original_name = source
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or("文件名无效")?
        .to_string();
    let id = Uuid::new_v4().to_string();
    let extension = source
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or("bin")
        .to_ascii_lowercase();
    let data_dir = app_data_dir(&app)?;
    let destination = data_dir.join("assets").join(format!("{id}.{extension}"));
    let temporary = data_dir.join("tmp").join(format!("{id}.importing"));
    fs::copy(&source, &temporary).map_err(|error| error.to_string())?;
    fs::rename(&temporary, &destination).map_err(|error| error.to_string())?;

    let sha256 = sha256_file(&destination)?;
    let mime_type = mime_guess::from_path(&source)
        .first_or_octet_stream()
        .to_string();
    let created_at = Utc::now().to_rfc3339();
    let connection = open_database(&app)?;
    if let Err(error) = connection.execute(
        "INSERT INTO assets(id, original_name, source_path, stored_path, mime_type, size_bytes, sha256, created_at)
         VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
        params![id, original_name, source.to_string_lossy(), destination.to_string_lossy(), mime_type, metadata.len(), sha256, created_at],
    ) {
        let _ = fs::remove_file(&destination);
        return Err(error.to_string());
    }

    Ok(AssetSummary {
        id,
        original_name,
        stored_path: destination.to_string_lossy().to_string(),
        mime_type,
        size_bytes: metadata.len(),
        created_at,
    })
}

#[tauri::command]
fn list_recent_assets(
    app: tauri::AppHandle,
    limit: Option<u32>,
) -> Result<Vec<AssetSummary>, String> {
    let connection = open_database(&app)?;
    let mut statement = connection
        .prepare(
            "SELECT id, original_name, stored_path, mime_type, size_bytes, created_at
             FROM assets WHERE deleted_at IS NULL ORDER BY created_at DESC LIMIT ?1",
        )
        .map_err(|error| error.to_string())?;
    let rows = statement
        .query_map([limit.unwrap_or(6).min(50)], |row| {
            Ok(AssetSummary {
                id: row.get(0)?,
                original_name: row.get(1)?,
                stored_path: row.get(2)?,
                mime_type: row.get(3)?,
                size_bytes: row.get(4)?,
                created_at: row.get(5)?,
            })
        })
        .map_err(|error| error.to_string())?;
    rows.collect::<Result<Vec<_>, _>>()
        .map_err(|error| error.to_string())
}

fn photo_source_by_id(
    connection: &Connection,
    source_id: &str,
) -> Result<Option<PhotoSourceSummary>, String> {
    connection
        .query_row(
            "SELECT id, path, label, recursive, status, photo_count, last_scanned_at
             FROM photo_sources WHERE id = ?1",
            [source_id],
            |row| {
                Ok(PhotoSourceSummary {
                    id: row.get(0)?,
                    path: row.get(1)?,
                    label: row.get(2)?,
                    recursive: row.get::<_, i64>(3)? != 0,
                    status: row.get(4)?,
                    photo_count: row.get(5)?,
                    last_scanned_at: row.get(6)?,
                })
            },
        )
        .optional()
        .map_err(|error| error.to_string())
}

#[tauri::command]
fn add_photo_source(app: tauri::AppHandle, path: String) -> Result<PhotoSourceSummary, String> {
    let canonical = PathBuf::from(path)
        .canonicalize()
        .map_err(|_| "无法读取所选文件夹".to_string())?;
    if !canonical.is_dir() {
        return Err("请选择一个可读取的文件夹".to_string());
    }
    let canonical_text = canonical.to_string_lossy().to_string();
    let label = canonical
        .file_name()
        .and_then(|value| value.to_str())
        .filter(|value| !value.is_empty())
        .unwrap_or(&canonical_text)
        .to_string();
    let connection = open_database(&app)?;
    if let Some(existing_id) = connection
        .query_row(
            "SELECT id FROM photo_sources WHERE path = ?1",
            [&canonical_text],
            |row| row.get::<_, String>(0),
        )
        .optional()
        .map_err(|error| error.to_string())?
    {
        return photo_source_by_id(&connection, &existing_id)?
            .ok_or_else(|| "相册来源状态无效".to_string());
    }
    let id = Uuid::new_v4().to_string();
    connection
        .execute(
            "INSERT INTO photo_sources(id, path, label, recursive, status, photo_count, created_at)
             VALUES(?1, ?2, ?3, 1, 'ready', 0, ?4)",
            params![id, canonical_text, label, Utc::now().to_rfc3339()],
        )
        .map_err(|error| error.to_string())?;
    photo_source_by_id(&connection, &id)?.ok_or_else(|| "无法保存相册来源".to_string())
}

#[tauri::command]
fn list_photo_sources(app: tauri::AppHandle) -> Result<Vec<PhotoSourceSummary>, String> {
    let connection = open_database(&app)?;
    let mut statement = connection
        .prepare(
            "SELECT id, path, label, recursive, status, photo_count, last_scanned_at
             FROM photo_sources ORDER BY created_at ASC",
        )
        .map_err(|error| error.to_string())?;
    let rows = statement
        .query_map([], |row| {
            Ok(PhotoSourceSummary {
                id: row.get(0)?,
                path: row.get(1)?,
                label: row.get(2)?,
                recursive: row.get::<_, i64>(3)? != 0,
                status: row.get(4)?,
                photo_count: row.get(5)?,
                last_scanned_at: row.get(6)?,
            })
        })
        .map_err(|error| error.to_string())?;
    rows.collect::<Result<Vec<_>, _>>()
        .map_err(|error| error.to_string())
}

#[tauri::command]
fn remove_photo_source(app: tauri::AppHandle, source_id: String) -> Result<(), String> {
    let connection = open_database(&app)?;
    let photo_ids = {
        let mut statement = connection
            .prepare("SELECT id FROM device_photos WHERE source_id = ?1")
            .map_err(|error| error.to_string())?;
        let ids = statement
            .query_map([&source_id], |row| row.get::<_, String>(0))
            .map_err(|error| error.to_string())?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| error.to_string())?;
        ids
    };
    let affected = connection
        .execute("DELETE FROM photo_sources WHERE id = ?1", [&source_id])
        .map_err(|error| error.to_string())?;
    if affected == 0 {
        return Err("相册来源不存在".to_string());
    }
    drop(connection);
    let thumbnail_dir = app_data_dir(&app)?.join("thumbnails");
    for photo_id in photo_ids {
        let _ = fs::remove_file(thumbnail_dir.join(format!("{photo_id}.jpg")));
        let _ = fs::remove_file(thumbnail_dir.join(format!("{photo_id}.writing")));
    }
    Ok(())
}

fn is_link_or_reparse_point(metadata: &fs::Metadata) -> bool {
    if metadata.file_type().is_symlink() {
        return true;
    }
    #[cfg(windows)]
    {
        const FILE_ATTRIBUTE_REPARSE_POINT: u32 = 0x0400;
        if metadata.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0 {
            return true;
        }
    }
    false
}

fn skip_scan_directory(path: &Path, metadata: &fs::Metadata) -> bool {
    #[cfg(windows)]
    {
        const FILE_ATTRIBUTE_HIDDEN: u32 = 0x0002;
        if metadata.file_attributes() & FILE_ATTRIBUTE_HIDDEN != 0 {
            return true;
        }
    }
    matches!(
        path.file_name()
            .and_then(|value| value.to_str())
            .map(str::to_ascii_lowercase)
            .as_deref(),
        Some("$recycle.bin" | "system volume information")
    ) || path
        .file_name()
        .and_then(|value| value.to_str())
        .is_some_and(|value| value.starts_with('.'))
}

fn create_local_thumbnail(source: &Path, destination: &Path) -> Result<(), String> {
    let dimensions = image::ImageReader::open(source)
        .map_err(|error| error.to_string())?
        .with_guessed_format()
        .map_err(|error| error.to_string())?
        .into_dimensions()
        .map_err(|error| error.to_string())?;
    let pixels = u64::from(dimensions.0)
        .checked_mul(u64::from(dimensions.1))
        .ok_or_else(|| "图片尺寸无效".to_string())?;
    if dimensions.0 == 0 || dimensions.1 == 0 || pixels > 40_000_000 {
        return Err("图片超过缩略图 4000 万像素安全上限".to_string());
    }
    let image = image::ImageReader::open(source)
        .map_err(|error| error.to_string())?
        .with_guessed_format()
        .map_err(|error| error.to_string())?
        .decode()
        .map_err(|error| error.to_string())?;
    let thumbnail = image.thumbnail(480, 480);
    let temporary = destination.with_extension("writing");
    let file = File::create(&temporary).map_err(|error| error.to_string())?;
    let mut writer = BufWriter::new(file);
    image::codecs::jpeg::JpegEncoder::new_with_quality(&mut writer, 82)
        .encode_image(&thumbnail)
        .map_err(|error| error.to_string())?;
    writer.flush().map_err(|error| error.to_string())?;
    fs::rename(&temporary, destination).map_err(|error| error.to_string())?;
    Ok(())
}

const TASK_CANCELLED: &str = "__task_cancelled__";

fn task_cancel_requested(app: &tauri::AppHandle, task_id: &str) -> Result<bool, String> {
    let connection = open_database(app)?;
    connection
        .query_row(
            "SELECT cancel_requested FROM tasks WHERE id = ?1",
            [task_id],
            |row| row.get::<_, i64>(0),
        )
        .map(|value| value != 0)
        .map_err(|error| error.to_string())
}

fn perform_photo_scan(
    app: &tauri::AppHandle,
    source_id: &str,
    task_id: &str,
) -> Result<PhotoSourceSummary, String> {
    let mut connection = open_database(&app)?;
    let source =
        photo_source_by_id(&connection, &source_id)?.ok_or_else(|| "相册来源不存在".to_string())?;
    let root = PathBuf::from(&source.path)
        .canonicalize()
        .map_err(|_| "相册来源当前不可访问".to_string())?;
    if !root.is_dir() {
        return Err("相册来源当前不是有效文件夹".to_string());
    }

    let scan_marker = Uuid::new_v4().to_string();
    let scanned_at = Utc::now().to_rfc3339();
    let mut stack = vec![root];
    let mut files = Vec::new();
    let mut visited_directories = 0_usize;
    const MAX_PHOTOS: usize = 50_000;
    const MAX_DIRECTORIES: usize = 20_000;

    while let Some(directory) = stack.pop() {
        visited_directories += 1;
        if visited_directories == 1 || visited_directories % 32 == 0 {
            if task_cancel_requested(app, task_id)? {
                return Err(TASK_CANCELLED.to_string());
            }
        }
        if visited_directories > MAX_DIRECTORIES {
            return Err(
                "目录数量超过本次 20000 个安全扫描上限，请选择更具体的照片文件夹".to_string(),
            );
        }
        let entries = fs::read_dir(&directory).map_err(|error| format!("无法读取目录：{error}"))?;
        for entry in entries {
            let entry = entry.map_err(|error| error.to_string())?;
            let path = entry.path();
            let metadata = fs::symlink_metadata(&path).map_err(|error| error.to_string())?;
            if is_link_or_reparse_point(&metadata) {
                continue;
            }
            if metadata.is_dir() {
                if source.recursive && !skip_scan_directory(&path, &metadata) {
                    stack.push(path);
                }
                continue;
            }
            if metadata.is_file() && allowed_extension(&path) {
                files.push((path, metadata));
                if files.len() > MAX_PHOTOS {
                    return Err(
                        "图片数量超过本次 50000 张安全扫描上限，请分开添加文件夹".to_string()
                    );
                }
            }
        }
    }

    if task_cancel_requested(app, task_id)? {
        return Err(TASK_CANCELLED.to_string());
    }

    let transaction = connection
        .transaction()
        .map_err(|error| error.to_string())?;
    transaction
        .execute(
            "UPDATE device_photos SET is_present = 0 WHERE source_id = ?1",
            [&source_id],
        )
        .map_err(|error| error.to_string())?;
    let mut thumbnail_jobs = Vec::new();
    for (path, metadata) in files {
        let path_text = path.to_string_lossy().to_string();
        let file_name = path
            .file_name()
            .and_then(|value| value.to_str())
            .unwrap_or("未命名图片")
            .to_string();
        let mime_type = mime_guess::from_path(&path)
            .first_or_octet_stream()
            .to_string();
        let modified_at = metadata
            .modified()
            .ok()
            .map(|value: SystemTime| chrono::DateTime::<Utc>::from(value).to_rfc3339());
        let now = Utc::now().to_rfc3339();
        let existing = transaction
            .query_row(
                "SELECT id, size_bytes, modified_at, thumbnail_path FROM device_photos
                 WHERE source_id = ?1 AND absolute_path = ?2",
                params![source_id, path_text],
                |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, u64>(1)?,
                        row.get::<_, Option<String>>(2)?,
                        row.get::<_, Option<String>>(3)?,
                    ))
                },
            )
            .optional()
            .map_err(|error| error.to_string())?;
        let photo_id = existing
            .as_ref()
            .map(|item| item.0.clone())
            .unwrap_or_else(|| Uuid::new_v4().to_string());
        let existing_thumbnail = existing.as_ref().and_then(|item| item.3.clone());
        let metadata_changed = existing
            .as_ref()
            .is_some_and(|item| item.1 != metadata.len() || item.2 != modified_at);
        let needs_thumbnail = metadata_changed
            || existing_thumbnail
                .as_ref()
                .is_none_or(|value| !Path::new(value).is_file());
        transaction
            .execute(
                "INSERT INTO device_photos(
                    id, source_id, absolute_path, file_name, mime_type, size_bytes, modified_at,
                    classification_status, thumbnail_status, is_present, scan_marker, created_at, updated_at
                 ) VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7, 'unclassified', 'pending', 1, ?8, ?9, ?9)
                 ON CONFLICT(source_id, absolute_path) DO UPDATE SET
                    file_name = excluded.file_name,
                    mime_type = excluded.mime_type,
                    size_bytes = excluded.size_bytes,
                    modified_at = excluded.modified_at,
                    category_id = CASE WHEN device_photos.classification_source = 'local_model'
                      AND (device_photos.size_bytes != excluded.size_bytes
                        OR COALESCE(device_photos.modified_at, '') != COALESCE(excluded.modified_at, ''))
                      THEN NULL ELSE device_photos.category_id END,
                    classification_status = CASE WHEN device_photos.classification_source = 'local_model'
                      AND (device_photos.size_bytes != excluded.size_bytes
                        OR COALESCE(device_photos.modified_at, '') != COALESCE(excluded.modified_at, ''))
                      THEN 'unclassified' ELSE device_photos.classification_status END,
                    classification_source = CASE WHEN device_photos.classification_source = 'local_model'
                      AND (device_photos.size_bytes != excluded.size_bytes
                        OR COALESCE(device_photos.modified_at, '') != COALESCE(excluded.modified_at, ''))
                      THEN 'none' ELSE device_photos.classification_source END,
                    classification_revision = CASE WHEN device_photos.classification_source = 'local_model'
                      AND (device_photos.size_bytes != excluded.size_bytes
                        OR COALESCE(device_photos.modified_at, '') != COALESCE(excluded.modified_at, ''))
                      THEN NULL ELSE device_photos.classification_revision END,
                    classification_label = CASE WHEN device_photos.classification_source = 'local_model'
                      AND (device_photos.size_bytes != excluded.size_bytes
                        OR COALESCE(device_photos.modified_at, '') != COALESCE(excluded.modified_at, ''))
                      THEN NULL ELSE device_photos.classification_label END,
                    thumbnail_status = CASE WHEN device_photos.size_bytes != excluded.size_bytes
                      OR COALESCE(device_photos.modified_at, '') != COALESCE(excluded.modified_at, '')
                      THEN 'pending' ELSE device_photos.thumbnail_status END,
                    is_present = 1,
                    scan_marker = excluded.scan_marker,
                    updated_at = excluded.updated_at",
                params![photo_id, source_id, path_text, file_name, mime_type, metadata.len(), modified_at, scan_marker, now],
            )
            .map_err(|error| error.to_string())?;
        if needs_thumbnail {
            thumbnail_jobs.push((photo_id, path));
        }
    }
    let photo_count: u64 = transaction
        .query_row(
            "SELECT COUNT(*) FROM device_photos WHERE source_id = ?1 AND is_present = 1",
            [&source_id],
            |row| row.get(0),
        )
        .map_err(|error| error.to_string())?;
    transaction
        .execute(
            "UPDATE photo_sources SET status = 'ready', photo_count = ?2, last_scanned_at = ?3 WHERE id = ?1",
            params![source_id, photo_count, scanned_at],
        )
        .map_err(|error| error.to_string())?;
    transaction.commit().map_err(|error| error.to_string())?;

    if !thumbnail_jobs.is_empty() {
        connection
            .execute(
                "UPDATE photo_sources SET status = 'thumbnailing' WHERE id = ?1",
                [source_id],
            )
            .map_err(|error| error.to_string())?;
    }
    let thumbnail_dir = app_data_dir(app)?.join("thumbnails");
    for (index, (photo_id, source_path)) in thumbnail_jobs.into_iter().enumerate() {
        if index % 8 == 0 && task_cancel_requested(app, task_id)? {
            return Err(TASK_CANCELLED.to_string());
        }
        let destination = thumbnail_dir.join(format!("{photo_id}.jpg"));
        let (status, stored_path) = match create_local_thumbnail(&source_path, &destination) {
            Ok(()) => ("ready", Some(destination.to_string_lossy().to_string())),
            Err(_) => {
                let _ = fs::remove_file(&destination);
                let _ = fs::remove_file(destination.with_extension("writing"));
                ("failed", None)
            }
        };
        connection
            .execute(
                "UPDATE device_photos SET thumbnail_status = ?2, thumbnail_path = ?3, updated_at = ?4 WHERE id = ?1",
                params![photo_id, status, stored_path, Utc::now().to_rfc3339()],
            )
            .map_err(|error| error.to_string())?;
    }
    connection
        .execute(
            "UPDATE photo_sources SET status = 'ready' WHERE id = ?1",
            [source_id],
        )
        .map_err(|error| error.to_string())?;
    photo_source_by_id(&connection, source_id)?.ok_or_else(|| "相册来源状态无效".to_string())
}

#[tauri::command]
fn start_photo_scan(app: tauri::AppHandle, source_id: String) -> Result<String, String> {
    let connection = open_database(&app)?;
    let source =
        photo_source_by_id(&connection, &source_id)?.ok_or_else(|| "相册来源不存在".to_string())?;
    if matches!(
        source.status.as_str(),
        "queued" | "scanning" | "thumbnailing"
    ) {
        return Err("该照片目录已经在扫描".to_string());
    }
    let task_id = Uuid::new_v4().to_string();
    let created_at = Utc::now().to_rfc3339();
    connection
        .execute(
            "INSERT INTO tasks(id, task_type, status, stage, parameters_json, created_at)
             VALUES(?1, 'album.scan', 'queued', 'enumerating', ?2, ?3)",
            params![
                task_id,
                serde_json::json!({"source_id": source_id}).to_string(),
                created_at
            ],
        )
        .map_err(|error| error.to_string())?;
    connection
        .execute(
            "UPDATE photo_sources SET status = 'queued' WHERE id = ?1",
            [&source_id],
        )
        .map_err(|error| error.to_string())?;
    drop(connection);

    let worker_app = app.clone();
    let worker_source_id = source_id.clone();
    let worker_task_id = task_id.clone();
    std::thread::spawn(move || {
        if let Ok(connection) = open_database(&worker_app) {
            let now = Utc::now().to_rfc3339();
            let _ = connection.execute(
                "UPDATE tasks SET status = 'running', stage = 'enumerating', started_at = ?2 WHERE id = ?1",
                params![worker_task_id, now],
            );
            let _ = connection.execute(
                "UPDATE photo_sources SET status = 'scanning' WHERE id = ?1",
                [&worker_source_id],
            );
        }
        match perform_photo_scan(&worker_app, &worker_source_id, &worker_task_id) {
            Ok(summary) => {
                if let Ok(connection) = open_database(&worker_app) {
                    let _ = connection.execute(
                        "UPDATE tasks SET status = 'succeeded', stage = 'complete', finished_at = ?2,
                         result_json = ?3 WHERE id = ?1",
                        params![worker_task_id, Utc::now().to_rfc3339(), serde_json::json!({"status":"pass","photo_count":summary.photo_count}).to_string()],
                    );
                }
            }
            Err(error) => {
                if let Ok(connection) = open_database(&worker_app) {
                    if error == TASK_CANCELLED {
                        let _ = connection.execute(
                            "UPDATE photo_sources SET status = 'interrupted' WHERE id = ?1",
                            [&worker_source_id],
                        );
                        let _ = connection.execute(
                            "UPDATE tasks SET status = 'cancelled', stage = 'cancelled', finished_at = ?2,
                             error_code = NULL WHERE id = ?1",
                            params![worker_task_id, Utc::now().to_rfc3339()],
                        );
                        return;
                    }
                    let _ = connection.execute(
                        "UPDATE photo_sources SET status = 'failed' WHERE id = ?1",
                        [&worker_source_id],
                    );
                    let _ = connection.execute(
                        "UPDATE tasks SET status = 'failed', stage = 'enumerating', finished_at = ?2,
                         error_code = 'album_scan_failed', result_json = ?3 WHERE id = ?1",
                        params![worker_task_id, Utc::now().to_rfc3339(), serde_json::json!({"status":"block","message":error}).to_string()],
                    );
                }
            }
        }
    });
    Ok(task_id)
}

fn request_task_cancellation(connection: &Connection, task_id: &str) -> Result<(), String> {
    let affected = connection
        .execute(
            "UPDATE tasks SET cancel_requested = 1
             WHERE id = ?1 AND status IN ('queued','running')",
            [task_id],
        )
        .map_err(|error| error.to_string())?;
    if affected == 0 {
        return Err("任务不存在、已经结束或无法取消".to_string());
    }
    Ok(())
}

#[tauri::command]
fn cancel_task(app: tauri::AppHandle, task_id: String) -> Result<(), String> {
    let connection = open_database(&app)?;
    request_task_cancellation(&connection, &task_id)
}

#[tauri::command]
fn cancel_photo_scan(app: tauri::AppHandle, source_id: String) -> Result<(), String> {
    let connection = open_database(&app)?;
    let mut statement = connection
        .prepare(
            "SELECT id, parameters_json FROM tasks
             WHERE task_type = 'album.scan' AND status IN ('queued','running')
             ORDER BY created_at DESC",
        )
        .map_err(|error| error.to_string())?;
    let rows = statement
        .query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })
        .map_err(|error| error.to_string())?;
    for row in rows {
        let (task_id, parameters) = row.map_err(|error| error.to_string())?;
        let matches_source = serde_json::from_str::<serde_json::Value>(&parameters)
            .ok()
            .and_then(|value| {
                value
                    .get("source_id")
                    .and_then(|item| item.as_str())
                    .map(|value| value == source_id)
            })
            .unwrap_or(false);
        if matches_source {
            return request_task_cancellation(&connection, &task_id);
        }
    }
    Err("该照片目录当前没有可取消的扫描任务".to_string())
}

#[tauri::command]
fn list_uncategorized_photos(
    app: tauri::AppHandle,
    limit: Option<u32>,
) -> Result<Vec<PhotoIndexSummary>, String> {
    let connection = open_database(&app)?;
    let mut statement = connection
        .prepare(
            "SELECT id, source_id, file_name, absolute_path, mime_type, size_bytes, modified_at,
                    category_id, classification_status, confidence, classification_source,
                    classification_label, thumbnail_path
             FROM device_photos
             WHERE is_present = 1 AND (category_id IS NULL OR classification_status = 'unclassified')
             ORDER BY COALESCE(modified_at, updated_at) DESC LIMIT ?1",
        )
        .map_err(|error| error.to_string())?;
    let rows = statement
        .query_map([limit.unwrap_or(200).min(1000)], |row| {
            Ok(PhotoIndexSummary {
                id: row.get(0)?,
                source_id: row.get(1)?,
                file_name: row.get(2)?,
                absolute_path: row.get(3)?,
                mime_type: row.get(4)?,
                size_bytes: row.get(5)?,
                modified_at: row.get(6)?,
                category_id: row.get(7)?,
                classification_status: row.get(8)?,
                confidence: row.get(9)?,
                classification_source: row.get(10)?,
                classification_label: row.get(11)?,
                thumbnail_path: row.get(12)?,
            })
        })
        .map_err(|error| error.to_string())?;
    rows.collect::<Result<Vec<_>, _>>()
        .map_err(|error| error.to_string())
}

#[tauri::command]
fn list_classified_photos(
    app: tauri::AppHandle,
    limit: Option<u32>,
) -> Result<Vec<PhotoIndexSummary>, String> {
    let connection = open_database(&app)?;
    let mut statement = connection
        .prepare(
            "SELECT id, source_id, file_name, absolute_path, mime_type, size_bytes, modified_at,
                    category_id, classification_status, confidence, classification_source,
                    classification_label, thumbnail_path
             FROM device_photos
             WHERE is_present = 1 AND category_id IS NOT NULL AND classification_status = 'classified'
             ORDER BY category_id ASC, COALESCE(modified_at, updated_at) DESC LIMIT ?1",
        )
        .map_err(|error| error.to_string())?;
    let rows = statement
        .query_map([limit.unwrap_or(500).min(2000)], |row| {
            Ok(PhotoIndexSummary {
                id: row.get(0)?,
                source_id: row.get(1)?,
                file_name: row.get(2)?,
                absolute_path: row.get(3)?,
                mime_type: row.get(4)?,
                size_bytes: row.get(5)?,
                modified_at: row.get(6)?,
                category_id: row.get(7)?,
                classification_status: row.get(8)?,
                confidence: row.get(9)?,
                classification_source: row.get(10)?,
                classification_label: row.get(11)?,
                thumbnail_path: row.get(12)?,
            })
        })
        .map_err(|error| error.to_string())?;
    rows.collect::<Result<Vec<_>, _>>()
        .map_err(|error| error.to_string())
}

#[tauri::command]
fn set_photo_category(
    app: tauri::AppHandle,
    photo_id: String,
    category_id: Option<String>,
) -> Result<(), String> {
    if let Some(category_id) = category_id.as_ref() {
        if !safe_category_ids().any(|known| known == category_id) {
            return Err("不支持该分类".to_string());
        }
    }
    let connection = open_database(&app)?;
    let status = if category_id.is_some() {
        "classified"
    } else {
        "unclassified"
    };
    let affected = connection
        .execute(
            "UPDATE device_photos SET category_id = ?2, classification_status = ?3,
             confidence = NULL, classification_source = 'manual', classification_revision = NULL,
             classification_label = NULL, updated_at = ?4 WHERE id = ?1 AND is_present = 1",
            params![photo_id, category_id, status, Utc::now().to_rfc3339()],
        )
        .map_err(|error| error.to_string())?;
    if affected == 0 {
        return Err("照片不存在或当前不可访问".to_string());
    }
    Ok(())
}

#[tauri::command]
fn set_photo_categories(
    app: tauri::AppHandle,
    photo_ids: Vec<String>,
    category_id: String,
) -> Result<u64, String> {
    if photo_ids.is_empty() || photo_ids.len() > 1000 {
        return Err("一次批量分类需要选择 1 至 1000 张图片".to_string());
    }
    if !safe_category_ids().any(|known| known == category_id) {
        return Err("不支持该分类".to_string());
    }
    let mut connection = open_database(&app)?;
    let transaction = connection
        .transaction()
        .map_err(|error| error.to_string())?;
    let now = Utc::now().to_rfc3339();
    let mut affected = 0_u64;
    for photo_id in photo_ids {
        affected += transaction
            .execute(
                "UPDATE device_photos SET category_id = ?2, classification_status = 'classified',
                 confidence = NULL, classification_source = 'manual', classification_revision = NULL,
                 classification_label = NULL, updated_at = ?3 WHERE id = ?1 AND is_present = 1",
                params![photo_id, category_id, now],
            )
            .map_err(|error| error.to_string())? as u64;
    }
    transaction.commit().map_err(|error| error.to_string())?;
    Ok(affected)
}

#[tauri::command]
fn import_device_photo(app: tauri::AppHandle, photo_id: String) -> Result<AssetSummary, String> {
    let connection = open_database(&app)?;
    let path = connection
        .query_row(
            "SELECT absolute_path FROM device_photos WHERE id = ?1 AND is_present = 1",
            [&photo_id],
            |row| row.get::<_, String>(0),
        )
        .optional()
        .map_err(|error| error.to_string())?
        .ok_or_else(|| "设备照片不存在或当前不可访问".to_string())?;
    drop(connection);
    import_asset(app, path)
}

#[tauri::command]
fn get_category_preferences(app: tauri::AppHandle) -> Result<Vec<CategoryPreference>, String> {
    let connection = open_database(&app)?;
    safe_categories()
        .iter()
        .map(|(id, label, description)| {
            let enabled = connection
                .query_row(
                    "SELECT enabled FROM category_preferences WHERE category_id = ?1",
                    [id],
                    |row| row.get::<_, i64>(0),
                )
                .map_err(|error| error.to_string())?
                != 0;
            Ok(CategoryPreference {
                id: (*id).to_string(),
                label: (*label).to_string(),
                description: (*description).to_string(),
                enabled,
            })
        })
        .collect()
}

#[tauri::command]
fn save_category_preferences(
    app: tauri::AppHandle,
    enabled_ids: Vec<String>,
) -> Result<Vec<CategoryPreference>, String> {
    if enabled_ids
        .iter()
        .any(|id| !safe_category_ids().any(|known| known == id))
    {
        return Err("包含不支持的分类".to_string());
    }
    let mut connection = open_database(&app)?;
    let transaction = connection
        .transaction()
        .map_err(|error| error.to_string())?;
    let now = Utc::now().to_rfc3339();
    for id in safe_category_ids() {
        transaction
            .execute(
                "INSERT INTO category_preferences(category_id, enabled, updated_at) VALUES(?1, ?2, ?3)
                 ON CONFLICT(category_id) DO UPDATE SET enabled = excluded.enabled, updated_at = excluded.updated_at",
                params![id, enabled_ids.iter().any(|enabled| enabled == id) as i64, now],
            )
            .map_err(|error| error.to_string())?;
    }
    transaction.commit().map_err(|error| error.to_string())?;
    get_category_preferences(app)
}

#[tauri::command]
fn start_album_classification(app: tauri::AppHandle) -> Result<String, String> {
    let connection = open_database(&app)?;
    let model_path = album_classifier_path(&app)?;
    if !model_path.is_file() {
        return Err("本地相册分类模型尚未安装，请先在模型中心安装".to_string());
    }
    let enabled_categories = {
        let mut statement = connection
            .prepare("SELECT category_id FROM category_preferences WHERE enabled = 1 ORDER BY category_id")
            .map_err(|error| error.to_string())?;
        let rows = statement
            .query_map([], |row| row.get::<_, String>(0))
            .map_err(|error| error.to_string())?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| error.to_string())?;
        rows
    };
    if enabled_categories.is_empty() {
        return Err("请先在分类设置中至少启用一个分类".to_string());
    }
    let active: u64 = connection
        .query_row(
            "SELECT COUNT(*) FROM tasks WHERE task_type = 'album.classify' AND status IN ('queued','running')",
            [],
            |row| row.get(0),
        )
        .map_err(|error| error.to_string())?;
    if active > 0 {
        return Err("本地自动分类任务已经在运行".to_string());
    }
    let items = {
        let mut statement = connection
            .prepare(
                "SELECT id, thumbnail_path FROM device_photos
                 WHERE is_present = 1 AND classification_status = 'unclassified'
                   AND classification_source != 'manual'
                   AND thumbnail_status = 'ready' AND thumbnail_path IS NOT NULL
                   AND COALESCE(classification_revision, '') != ?1
                 ORDER BY updated_at ASC LIMIT 256",
            )
            .map_err(|error| error.to_string())?;
        let rows = statement
            .query_map([ALBUM_CLASSIFIER_REVISION], |row| {
                Ok(serde_json::json!({
                    "id": row.get::<_, String>(0)?,
                    "path": row.get::<_, String>(1)?,
                }))
            })
            .map_err(|error| error.to_string())?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| error.to_string())?;
        rows
    };
    if items.is_empty() {
        return Err("当前没有需要自动分类的新图片".to_string());
    }

    let task_id = Uuid::new_v4().to_string();
    let created_at = Utc::now().to_rfc3339();
    connection
        .execute(
            "INSERT INTO tasks(id, task_type, status, stage, parameters_json, created_at)
             VALUES(?1, 'album.classify', 'queued', 'loading_model', ?2, ?3)",
            params![
                task_id,
                serde_json::json!({
                    "revision": ALBUM_CLASSIFIER_REVISION,
                    "photo_count": items.len(),
                    "enabled_categories": enabled_categories,
                })
                .to_string(),
                created_at
            ],
        )
        .map_err(|error| error.to_string())?;
    drop(connection);

    let worker_app = app.clone();
    let worker_task_id = task_id.clone();
    std::thread::spawn(move || {
        if let Ok(connection) = open_database(&worker_app) {
            let _ = connection.execute(
                "UPDATE tasks SET status = 'running', stage = 'classifying', started_at = ?2 WHERE id = ?1",
                params![worker_task_id, Utc::now().to_rfc3339()],
            );
        }
        if task_cancel_requested(&worker_app, &worker_task_id).unwrap_or(false) {
            if let Ok(connection) = open_database(&worker_app) {
                let _ = connection.execute(
                    "UPDATE tasks SET status = 'cancelled', stage = 'cancelled', finished_at = ?2 WHERE id = ?1",
                    params![worker_task_id, Utc::now().to_rfc3339()],
                );
            }
            return;
        }
        let execution = call_engine(
            "album.classify_safe_batch",
            serde_json::json!({
                "model_path": model_path.to_string_lossy(),
                "enabled_categories": enabled_categories,
                "items": items,
            }),
        );
        match execution {
            Ok((_, payload)) => {
                if task_cancel_requested(&worker_app, &worker_task_id).unwrap_or(false) {
                    if let Ok(connection) = open_database(&worker_app) {
                        let _ = connection.execute(
                            "UPDATE tasks SET status = 'cancelled', stage = 'cancelled', finished_at = ?2 WHERE id = ?1",
                            params![worker_task_id, Utc::now().to_rfc3339()],
                        );
                    }
                    return;
                }
                let results = payload
                    .get("results")
                    .and_then(|value| value.as_array())
                    .cloned()
                    .unwrap_or_default();
                if let Ok(mut connection) = open_database(&worker_app) {
                    let mut classified = 0_u64;
                    let mut uncategorized = 0_u64;
                    let mut failed = 0_u64;
                    if let Ok(transaction) = connection.transaction() {
                        for result in results {
                            let Some(photo_id) = result.get("id").and_then(|value| value.as_str())
                            else {
                                failed += 1;
                                continue;
                            };
                            let category_id =
                                result.get("category_id").and_then(|value| value.as_str());
                            let confidence =
                                result.get("confidence").and_then(|value| value.as_f64());
                            let label = result.get("label").and_then(|value| value.as_str());
                            if result.get("error").is_some_and(|value| !value.is_null()) {
                                failed += 1;
                            } else if category_id.is_some() {
                                classified += 1;
                            } else {
                                uncategorized += 1;
                            }
                            let status = if category_id.is_some() {
                                "classified"
                            } else {
                                "unclassified"
                            };
                            let _ = transaction.execute(
                                "UPDATE device_photos SET category_id = ?2, classification_status = ?3,
                                 confidence = ?4, classification_source = 'local_model',
                                 classification_revision = ?5, classification_label = ?6, updated_at = ?7
                                 WHERE id = ?1 AND is_present = 1 AND classification_source != 'manual'",
                                params![photo_id, category_id, status, confidence, ALBUM_CLASSIFIER_REVISION, label, Utc::now().to_rfc3339()],
                            );
                        }
                        let result_json = serde_json::json!({
                            "status": "pass",
                            "classified": classified,
                            "uncategorized": uncategorized,
                            "failed": failed,
                            "revision": ALBUM_CLASSIFIER_REVISION,
                            "device": payload.get("device"),
                        });
                        let _ = transaction.execute(
                            "UPDATE tasks SET status = 'succeeded', stage = 'complete', finished_at = ?2,
                             result_json = ?3 WHERE id = ?1",
                            params![worker_task_id, Utc::now().to_rfc3339(), result_json.to_string()],
                        );
                        let _ = transaction.commit();
                    }
                }
            }
            Err(error) => {
                if let Ok(connection) = open_database(&worker_app) {
                    let _ = connection.execute(
                        "UPDATE tasks SET status = 'failed', stage = 'classifying', finished_at = ?2,
                         error_code = 'album_classification_failed', result_json = ?3 WHERE id = ?1",
                        params![worker_task_id, Utc::now().to_rfc3339(), serde_json::json!({"status":"block","message":error}).to_string()],
                    );
                }
            }
        }
    });
    Ok(task_id)
}

#[tauri::command]
fn list_tasks(app: tauri::AppHandle, limit: Option<u32>) -> Result<Vec<TaskSummary>, String> {
    let connection = open_database(&app)?;
    let mut statement = connection
        .prepare(
            "SELECT id, task_type, status, stage, created_at, finished_at, error_code, result_json,
                    cancel_requested
             FROM tasks ORDER BY created_at DESC LIMIT ?1",
        )
        .map_err(|error| error.to_string())?;
    let rows = statement
        .query_map([limit.unwrap_or(30).min(100)], |row| {
            let result_json = row.get::<_, Option<String>>(7)?;
            let result_status = result_json
                .and_then(|value| serde_json::from_str::<serde_json::Value>(&value).ok())
                .and_then(|value| {
                    value
                        .get("status")
                        .and_then(|item| item.as_str())
                        .map(str::to_string)
                });
            Ok(TaskSummary {
                id: row.get(0)?,
                task_type: row.get(1)?,
                status: row.get(2)?,
                stage: row.get(3)?,
                created_at: row.get(4)?,
                finished_at: row.get(5)?,
                error_code: row.get(6)?,
                result_status,
                cancel_requested: row.get::<_, i64>(8)? != 0,
            })
        })
        .map_err(|error| error.to_string())?;
    rows.collect::<Result<Vec<_>, _>>()
        .map_err(|error| error.to_string())
}

fn asset_summary_by_id(
    connection: &Connection,
    asset_id: &str,
) -> Result<Option<AssetSummary>, String> {
    connection
        .query_row(
            "SELECT id, original_name, stored_path, mime_type, size_bytes, created_at
             FROM assets WHERE id = ?1 AND deleted_at IS NULL",
            [asset_id],
            |row| {
                Ok(AssetSummary {
                    id: row.get(0)?,
                    original_name: row.get(1)?,
                    stored_path: row.get(2)?,
                    mime_type: row.get(3)?,
                    size_bytes: row.get(4)?,
                    created_at: row.get(5)?,
                })
            },
        )
        .optional()
        .map_err(|error| error.to_string())
}

#[tauri::command]
fn load_tryon_draft(app: tauri::AppHandle) -> Result<TryOnDraft, String> {
    let connection = open_database(&app)?;
    let stored = connection
        .query_row(
            "SELECT person_asset_id, garment_asset_id, category, garment_photo_type, sleeve_mode, updated_at
             FROM tryon_drafts WHERE id = 'current'",
            [],
            |row| {
                Ok((
                    row.get::<_, Option<String>>(0)?,
                    row.get::<_, Option<String>>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, String>(3)?,
                    row.get::<_, String>(4)?,
                    row.get::<_, String>(5)?,
                ))
            },
        )
        .optional()
        .map_err(|error| error.to_string())?;

    let Some((person_id, garment_id, category, garment_photo_type, sleeve_mode, updated_at)) = stored else {
        return Ok(TryOnDraft {
            person: None,
            garment: None,
            category: "automatic".to_string(),
            garment_photo_type: "model".to_string(),
            sleeve_mode: "automatic".to_string(),
            updated_at: None,
        });
    };
    let person = match person_id {
        Some(id) => asset_summary_by_id(&connection, &id)?,
        None => None,
    };
    let garment = match garment_id {
        Some(id) => asset_summary_by_id(&connection, &id)?,
        None => None,
    };
    Ok(TryOnDraft {
        person,
        garment,
        category,
        garment_photo_type,
        sleeve_mode,
        updated_at: Some(updated_at),
    })
}

fn validate_draft_asset(connection: &Connection, asset_id: Option<&String>) -> Result<(), String> {
    if let Some(asset_id) = asset_id {
        if asset_summary_by_id(connection, asset_id)?.is_none() {
            return Err("穿搭草稿引用的素材不存在或已删除".to_string());
        }
    }
    Ok(())
}

#[tauri::command]
fn save_tryon_draft(
    app: tauri::AppHandle,
    person_asset_id: Option<String>,
    garment_asset_id: Option<String>,
    category: String,
    garment_photo_type: String,
    sleeve_mode: String,
) -> Result<TryOnDraft, String> {
    if !matches!(category.as_str(), "automatic" | "tops" | "bottoms" | "one-pieces") {
        return Err("服装类别无效".to_string());
    }
    if !matches!(garment_photo_type.as_str(), "model" | "flat-lay") {
        return Err("服装照片类型无效".to_string());
    }
    if !matches!(sleeve_mode.as_str(), "automatic" | "standard" | "reveal-arms") {
        return Err("袖长与遮挡模式无效".to_string());
    }
    let connection = open_database(&app)?;
    validate_draft_asset(&connection, person_asset_id.as_ref())?;
    validate_draft_asset(&connection, garment_asset_id.as_ref())?;
    let updated_at = Utc::now().to_rfc3339();
    connection
        .execute(
            "INSERT INTO tryon_drafts(id, person_asset_id, garment_asset_id, category, garment_photo_type, sleeve_mode, updated_at)
             VALUES('current', ?1, ?2, ?3, ?4, ?5, ?6)
             ON CONFLICT(id) DO UPDATE SET
               person_asset_id = excluded.person_asset_id,
               garment_asset_id = excluded.garment_asset_id,
               category = excluded.category,
               garment_photo_type = excluded.garment_photo_type,
               sleeve_mode = excluded.sleeve_mode,
               updated_at = excluded.updated_at",
            params![person_asset_id, garment_asset_id, category, garment_photo_type, sleeve_mode, updated_at],
        )
        .map_err(|error| error.to_string())?;
    drop(connection);
    load_tryon_draft(app)
}

fn project_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap_or(Path::new("."))
        .to_path_buf()
}

#[tauri::command]
fn check_engine_health() -> EngineHealth {
    match call_engine("health", serde_json::json!({})) {
        Ok((protocol_version, payload)) => EngineHealth {
            available: true,
            python_version: payload
                .get("python_version")
                .and_then(|value| value.as_str())
                .map(str::to_string),
            platform: payload
                .get("platform")
                .and_then(|value| value.as_str())
                .map(str::to_string),
            protocol_version: Some(protocol_version),
            message: "桌面端已连接本地 Python 引擎".to_string(),
        },
        Err(error) => EngineHealth {
            available: false,
            python_version: None,
            platform: None,
            protocol_version: None,
            message: error,
        },
    }
}

fn call_engine(
    operation: &str,
    payload: serde_json::Value,
) -> Result<(u64, serde_json::Value), String> {
    let worker = project_root().join("engine").join("worker.py");
    let mut child = match Command::new("python")
        .arg(worker)
        .arg("--once")
        .env("PYTHONIOENCODING", "utf-8")
        .env("PYTHONUTF8", "1")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(child) => child,
        Err(error) => return Err(format!("无法启动 Python 引擎：{error}")),
    };

    if let Some(mut stdin) = child.stdin.take() {
        let request = serde_json::json!({
            "protocol_version": 1,
            "request_id": Uuid::new_v4(),
            "op": operation,
            "payload": payload,
        });
        if writeln!(stdin, "{request}").is_err() {
            return Err("无法写入引擎请求".to_string());
        }
    }

    match child.wait_with_output() {
        Ok(output) if output.status.success() => {
            match serde_json::from_slice::<EngineEnvelope>(&output.stdout) {
                Ok(response) if response.success => response
                    .payload
                    .map(|payload| (response.protocol_version, payload))
                    .ok_or_else(|| "引擎响应缺少 payload".to_string()),
                Ok(response) => {
                    let error = response
                        .error
                        .map(|error| format!("{}：{}", error.code, error.message));
                    Err(error.unwrap_or_else(|| "引擎返回失败状态".to_string()))
                }
                Err(error) => Err(format!("无法解析引擎响应：{error}")),
            }
        }
        Ok(output) => Err(String::from_utf8_lossy(&output.stderr).trim().to_string()),
        Err(error) => Err(format!("等待引擎失败：{error}")),
    }
}

fn tryon_weights_dir(app: &tauri::AppHandle) -> Result<PathBuf, String> {
    let installed = app_data_dir(app)?
        .join("models")
        .join("tryon")
        .join("weights");
    if installed.join("model.safetensors").is_file() {
        return Ok(installed);
    }
    Ok(project_root().join("models").join("tryon").join("weights"))
}

fn tryon_runtime_python() -> PathBuf {
    project_root()
        .join("runtime")
        .join("tryon")
        .join(".venv")
        .join("Scripts")
        .join("python.exe")
}

fn tryon_human_parser_dir() -> PathBuf {
    project_root()
        .join("runtime")
        .join("tryon")
        .join("hf-cache")
        .join("hub")
        .join("models--fashn-ai--fashn-human-parser")
        .join("snapshots")
        .join("1f80c34dbab321c5730dda5c3fea279fd3e97498")
}

fn call_tryon_engine(
    _app: &tauri::AppHandle,
    operation: &str,
    payload: serde_json::Value,
) -> Result<serde_json::Value, String> {
    let python = tryon_runtime_python();
    if !python.is_file() {
        return Err("本地试衣隔离运行时尚未安装".to_string());
    }
    let root = project_root();
    let worker = root.join("engine").join("tryon_worker.py");
    let hf_home = root.join("runtime").join("tryon").join("hf-cache");
    let mut child = Command::new(python)
        .arg(worker)
        .current_dir(&root)
        .env("HF_HOME", hf_home)
        .env("HF_HUB_OFFLINE", "1")
        .env("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        .env("PYTHONIOENCODING", "utf-8")
        .env("PYTHONUTF8", "1")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| format!("无法启动本地试衣运行时：{error}"))?;

    if let Some(mut stdin) = child.stdin.take() {
        let request = serde_json::json!({
            "protocol_version": 1,
            "request_id": Uuid::new_v4(),
            "op": operation,
            "payload": payload,
        });
        writeln!(stdin, "{request}").map_err(|_| "无法写入试衣引擎请求".to_string())?;
    }

    let output = child
        .wait_with_output()
        .map_err(|error| format!("等待试衣引擎失败：{error}"))?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let response_line = stdout
        .lines()
        .rev()
        .find(|line| !line.trim().is_empty())
        .ok_or_else(|| {
            let detail = String::from_utf8_lossy(&output.stderr);
            format!("试衣引擎没有返回结果：{}", detail.trim())
        })?;
    let response = serde_json::from_str::<EngineEnvelope>(response_line)
        .map_err(|error| format!("无法解析试衣引擎响应：{error}"))?;
    if !response.success {
        return Err(response
            .error
            .map(|error| format!("{}：{}", error.code, error.message))
            .unwrap_or_else(|| "试衣引擎返回失败状态".to_string()));
    }
    response
        .payload
        .ok_or_else(|| "试衣引擎响应缺少 payload".to_string())
}

#[tauri::command]
fn get_tryon_model_status(app: tauri::AppHandle) -> Result<serde_json::Value, String> {
    let weights_dir = tryon_weights_dir(&app)?;
    let human_parser_dir = tryon_human_parser_dir();
    let device = detect_device();
    let result = call_tryon_engine(
        &app,
        "tryon.status",
        serde_json::json!({
            "weights_dir": weights_dir,
            "human_parser_dir": human_parser_dir,
            "cuda_available": device.gpu_available,
            "gpu_name": device.gpu_name,
        }),
    )?;
    let ready = result
        .get("ready")
        .and_then(|value| value.as_bool())
        .unwrap_or(false);
    let connection = open_database(&app)?;
    if ready {
        connection
            .execute(
                "INSERT INTO installed_models(id, revision, status, installed_at)
                 VALUES('fashn-vton-1.5', ?1, 'available', ?2)
                 ON CONFLICT(id) DO UPDATE SET revision = excluded.revision, status = excluded.status",
                params![
                    result
                        .get("model_revision")
                        .and_then(|value| value.as_str())
                        .unwrap_or("unknown"),
                    Utc::now().to_rfc3339()
                ],
            )
            .map_err(|error| error.to_string())?;
    } else {
        connection
            .execute("DELETE FROM installed_models WHERE id = 'fashn-vton-1.5'", [])
            .map_err(|error| error.to_string())?;
    }
    Ok(result)
}

fn hand_refiner_runtime_root() -> PathBuf {
    project_root().join("runtime").join("hand-refiner")
}

fn hand_refiner_weights_root() -> PathBuf {
    project_root().join("models").join("hand-refiner").join("weights")
}

fn call_hand_refiner_engine(
    operation: &str,
    payload: serde_json::Value,
) -> Result<serde_json::Value, String> {
    let root = project_root();
    let runtime_root = hand_refiner_runtime_root();
    let isolated_python = runtime_root.join(".venv").join("Scripts").join("python.exe");
    let worker = root.join("engine").join("hand_refiner_worker.py");
    let mut command = if isolated_python.is_file() {
        Command::new(&isolated_python)
    } else {
        Command::new("python")
    };
    let mut child = command
        .arg(worker)
        .current_dir(&root)
        .env("PYTHONIOENCODING", "utf-8")
        .env("PYTHONUTF8", "1")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| format!("无法检查手部修复运行时：{error}"))?;
    if let Some(mut stdin) = child.stdin.take() {
        let request = serde_json::json!({
            "protocol_version": 1,
            "request_id": Uuid::new_v4(),
            "op": operation,
            "payload": payload,
        });
        writeln!(stdin, "{request}").map_err(|_| "无法写入手部修复状态请求".to_string())?;
    }
    let output = child
        .wait_with_output()
        .map_err(|error| format!("等待手部修复状态失败：{error}"))?;
    let response = serde_json::from_slice::<EngineEnvelope>(&output.stdout).map_err(|error| {
        let detail = String::from_utf8_lossy(&output.stderr);
        format!("无法解析手部修复状态：{error}；{}", detail.trim())
    })?;
    if !response.success {
        return Err(response
            .error
            .map(|error| format!("{}：{}", error.code, error.message))
            .unwrap_or_else(|| "手部修复状态检查失败".to_string()));
    }
    response.payload.ok_or_else(|| "手部修复状态缺少 payload".to_string())
}

#[tauri::command]
fn get_hand_refiner_status() -> Result<serde_json::Value, String> {
    call_hand_refiner_engine(
        "hand_refiner.status",
        serde_json::json!({
            "runtime_root": hand_refiner_runtime_root(),
            "weights_root": hand_refiner_weights_root(),
        }),
    )
}

#[tauri::command]
fn run_tryon_generation(
    app: tauri::AppHandle,
    person_asset_id: String,
    garment_asset_id: String,
    category: String,
    garment_photo_type: String,
    sleeve_mode: String,
    num_timesteps: u32,
) -> Result<serde_json::Value, String> {
    if !matches!(category.as_str(), "automatic" | "tops" | "bottoms" | "one-pieces") {
        return Err("服装类别无效".to_string());
    }
    if !matches!(garment_photo_type.as_str(), "model" | "flat-lay") {
        return Err("服装照片类型无效".to_string());
    }
    if !matches!(sleeve_mode.as_str(), "automatic" | "standard" | "reveal-arms") {
        return Err("袖长与遮挡模式无效".to_string());
    }
    if !matches!(num_timesteps, 20 | 30) {
        return Err("生成步数只允许 20 或 30".to_string());
    }

    let connection = open_database(&app)?;
    let person = asset_summary_by_id(&connection, &person_asset_id)?
        .ok_or_else(|| "人物素材不存在或已删除".to_string())?;
    let garment = asset_summary_by_id(&connection, &garment_asset_id)?
        .ok_or_else(|| "服装素材不存在或已删除".to_string())?;
    let task_id = Uuid::new_v4().to_string();
    let result_asset_id = Uuid::new_v4().to_string();
    let created_at = Utc::now().to_rfc3339();
    let seed = 42_u32;
    let parameters = serde_json::json!({
        "person_asset_id": &person_asset_id,
        "garment_asset_id": &garment_asset_id,
        "category": &category,
        "garment_photo_type": &garment_photo_type,
        "sleeve_mode": &sleeve_mode,
        "num_timesteps": num_timesteps,
        "seed": seed,
    });
    connection
        .execute(
            "INSERT INTO tasks(id, task_type, status, stage, parameters_json, created_at, started_at)
             VALUES(?1, 'tryon.generate', 'running', 'input_checks', ?2, ?3, ?3)",
            params![task_id, parameters.to_string(), created_at],
        )
        .map_err(|error| error.to_string())?;
    drop(connection);

    let data_dir = app_data_dir(&app)?;
    let output_path = data_dir
        .join("assets")
        .join(format!("{result_asset_id}.png"));
    let execution = (|| -> Result<serde_json::Value, String> {
        let (_, person_check) = call_engine(
            "image.preflight",
            serde_json::json!({ "path": &person.stored_path, "role": "person" }),
        )?;
        let (_, garment_check) = call_engine(
            "image.preflight",
            serde_json::json!({ "path": &garment.stored_path, "role": "garment" }),
        )?;
        if preflight_status(&person_check) == "block"
            || preflight_status(&garment_check) == "block"
        {
            return Err("输入预检存在阻止项，请更换图片后重试".to_string());
        }
        if task_cancel_requested(&app, &task_id)? {
            return Err("任务已取消".to_string());
        }
        open_database(&app)?
            .execute(
                "UPDATE tasks SET stage = 'local_generation' WHERE id = ?1",
                [&task_id],
            )
            .map_err(|error| error.to_string())?;
        let weights_dir = tryon_weights_dir(&app)?;
        let human_parser_dir = tryon_human_parser_dir();
        let mut result = call_tryon_engine(
            &app,
            "tryon.generate",
            serde_json::json!({
                "person_path": &person.stored_path,
                "garment_path": &garment.stored_path,
                "output_path": &output_path,
                "weights_dir": &weights_dir,
                "human_parser_dir": &human_parser_dir,
                "category": &category,
                "garment_photo_type": &garment_photo_type,
                "sleeve_mode": &sleeve_mode,
                "num_timesteps": num_timesteps,
                "seed": seed,
            }),
        )?;
        let missing_hands: Vec<String> = result
            .pointer("/quality_checks/limbs/repair_targets")
            .and_then(|value| value.as_array())
            .map(|targets| {
                targets
                    .iter()
                    .filter_map(|value| value.as_str())
                    .filter(|target| target.starts_with("hand_"))
                    .map(str::to_string)
                    .collect()
            })
            .unwrap_or_default();
        if !missing_hands.is_empty() {
            let refiner_ready = get_hand_refiner_status()
                .ok()
                .and_then(|status| status.get("ready").and_then(|value| value.as_bool()))
                .unwrap_or(false);
            if refiner_ready {
                open_database(&app)?
                    .execute(
                        "UPDATE tasks SET stage = 'hand_refinement' WHERE id = ?1",
                        [&task_id],
                    )
                    .map_err(|error| error.to_string())?;
                let cache_root = data_dir.join("cache").join("hand-repair");
                let repair_dir = cache_root.join(&task_id);
                fs::create_dir_all(&repair_dir).map_err(|error| error.to_string())?;
                let mut repair_groups = Vec::new();
                let mut hand_generation_elapsed = 0.0_f64;
                for target in &missing_hands {
                    let side = target.trim_start_matches("hand_");
                    let repair_path = result
                        .pointer(&format!("/quality_checks/limbs/repair_paths/{target}"))
                        .cloned();
                    let Some(repair_path) = repair_path else {
                        continue;
                    };
                    if let Ok(group) = call_hand_refiner_engine(
                        "hand_refiner.generate",
                        serde_json::json!({
                            "runtime_root": hand_refiner_runtime_root(),
                            "weights_root": hand_refiner_weights_root(),
                            "image_path": &output_path,
                            "output_dir": repair_dir.join(side),
                            "repair_path": repair_path,
                            "side": side,
                            "seeds": [839, 173, 613, 727],
                            "steps": 30,
                        }),
                    ) {
                        hand_generation_elapsed += group
                            .get("elapsed_seconds")
                            .and_then(|value| value.as_f64())
                            .unwrap_or(0.0);
                        repair_groups.push(group);
                    }
                }
                if !repair_groups.is_empty() {
                    if let Ok(finalized) = call_tryon_engine(
                        &app,
                        "tryon.finalize_hand_candidates",
                        serde_json::json!({
                            "person_path": &person.stored_path,
                            "garment_path": &garment.stored_path,
                            "output_path": &output_path,
                            "weights_dir": &weights_dir,
                            "human_parser_dir": &human_parser_dir,
                            "category": result.get("category").and_then(|value| value.as_str()).unwrap_or(&category),
                            "garment_photo_type": &garment_photo_type,
                            "sleeve_mode": &sleeve_mode,
                            "repair_groups": repair_groups,
                        }),
                    ) {
                        let repair_applied = finalized
                            .get("repair_applied")
                            .and_then(|value| value.as_bool())
                            .unwrap_or(false);
                        if let Some(object) = result.as_object_mut() {
                            for key in ["verification_status", "quality_checks"] {
                                if let Some(value) = finalized.get(key) {
                                    object.insert(key.to_string(), value.clone());
                                }
                            }
                            if let Some(value) = finalized.pointer("/quality_checks/skin_tone/delta_before") {
                                object.insert("skin_tone_delta_before".to_string(), value.clone());
                            }
                            if let Some(value) = finalized.pointer("/quality_checks/skin_tone/delta_after") {
                                object.insert("skin_tone_delta_after".to_string(), value.clone());
                            }
                            let elapsed = object
                                .get("elapsed_seconds")
                                .and_then(|value| value.as_f64())
                                .unwrap_or(0.0)
                                + hand_generation_elapsed
                                + finalized
                                    .get("elapsed_seconds")
                                    .and_then(|value| value.as_f64())
                                    .unwrap_or(0.0);
                            object.insert("elapsed_seconds".to_string(), serde_json::json!(elapsed));
                            object.insert("hand_refinement_used".to_string(), serde_json::json!(repair_applied));
                            object.insert(
                                "hand_refinement_revision".to_string(),
                                serde_json::json!("sd15-hand-inpaint-local-v1"),
                            );
                            let existing_notice = object
                                .get("coverage_notice")
                                .and_then(|value| value.as_str())
                                .unwrap_or("");
                            let hand_notice = if repair_applied {
                                " 独立手部模型已生成多个候选，并仅合并通过左右手、肤色和服装复核的结果。"
                            } else {
                                " 独立手部候选均未通过完整复核，已保留手部修复前结果。"
                            };
                            object.insert(
                                "coverage_notice".to_string(),
                                serde_json::json!(format!("{existing_notice}{hand_notice}")),
                            );
                        }
                    }
                }
                if repair_dir.starts_with(&cache_root) {
                    let _ = fs::remove_dir_all(&repair_dir);
                }
            }
        }
        if task_cancel_requested(&app, &task_id)? {
            let _ = fs::remove_file(&output_path);
            return Err("任务已取消".to_string());
        }

        let metadata = output_path.metadata().map_err(|error| error.to_string())?;
        let sha256 = sha256_file(&output_path)?;
        let finished_at = Utc::now().to_rfc3339();
        let original_name = format!("虚拟穿搭-{}.png", &task_id[..8]);
        let asset = AssetSummary {
            id: result_asset_id.clone(),
            original_name: original_name.clone(),
            stored_path: output_path.to_string_lossy().to_string(),
            mime_type: "image/png".to_string(),
            size_bytes: metadata.len(),
            created_at: finished_at.clone(),
        };
        let connection = open_database(&app)?;
        connection
            .execute(
                "INSERT INTO assets(id, original_name, source_path, stored_path, mime_type, size_bytes, sha256, created_at)
                 VALUES(?1, ?2, 'generated:tryon', ?3, 'image/png', ?4, ?5, ?6)",
                params![result_asset_id, original_name, output_path.to_string_lossy(), metadata.len(), sha256, finished_at],
            )
            .map_err(|error| error.to_string())?;
        if let Some(object) = result.as_object_mut() {
            object.insert("task_id".to_string(), serde_json::json!(&task_id));
            object.insert("asset".to_string(), serde_json::json!(asset));
        }
        Ok(result)
    })();

    let connection = open_database(&app)?;
    match execution {
        Ok(result) => {
            connection
                .execute(
                    "UPDATE tasks SET status = 'succeeded', stage = 'complete', finished_at = ?2, result_json = ?3 WHERE id = ?1",
                    params![task_id, Utc::now().to_rfc3339(), result.to_string()],
                )
                .map_err(|error| error.to_string())?;
            Ok(result)
        }
        Err(error) => {
            let cancelled = error == "任务已取消";
            let error_code = if cancelled {
                None
            } else if error.contains("gpu_out_of_memory") {
                Some("gpu_out_of_memory")
            } else {
                Some("tryon_failed")
            };
            connection
                .execute(
                    "UPDATE tasks SET status = ?2, stage = ?3, finished_at = ?4, error_code = ?5 WHERE id = ?1",
                    params![
                        task_id,
                        if cancelled { "cancelled" } else { "failed" },
                        if cancelled { "cancelled" } else { "local_generation" },
                        Utc::now().to_rfc3339(),
                        error_code,
                    ],
                )
                .map_err(|database_error| database_error.to_string())?;
            let _ = fs::remove_file(&output_path);
            Err(error)
        }
    }
}

#[tauri::command]
fn list_tryon_results(
    app: tauri::AppHandle,
    limit: Option<u32>,
) -> Result<Vec<serde_json::Value>, String> {
    let connection = open_database(&app)?;
    let rows = {
        let mut statement = connection
            .prepare(
                "SELECT id, created_at, parameters_json, result_json
                 FROM tasks
                 WHERE task_type = 'tryon.generate' AND status = 'succeeded' AND result_json IS NOT NULL
                 ORDER BY created_at DESC LIMIT ?1",
            )
            .map_err(|error| error.to_string())?;
        let mapped = statement
            .query_map([limit.unwrap_or(20).min(50)], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, String>(3)?,
                ))
            })
            .map_err(|error| error.to_string())?;
        mapped
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| error.to_string())?
    };

    let mut results = Vec::new();
    for (task_id, created_at, parameters_json, result_json) in rows {
        let parameters = match serde_json::from_str::<serde_json::Value>(&parameters_json) {
            Ok(value) => value,
            Err(_) => continue,
        };
        let mut result = match serde_json::from_str::<serde_json::Value>(&result_json) {
            Ok(value) => value,
            Err(_) => continue,
        };
        let result_asset_id = result
            .get("asset")
            .and_then(|value| value.get("id"))
            .and_then(|value| value.as_str());
        let person_asset_id = parameters
            .get("person_asset_id")
            .and_then(|value| value.as_str());
        let garment_asset_id = parameters
            .get("garment_asset_id")
            .and_then(|value| value.as_str());
        let (Some(result_asset_id), Some(person_asset_id), Some(garment_asset_id)) =
            (result_asset_id, person_asset_id, garment_asset_id)
        else {
            continue;
        };
        let Some(asset) = asset_summary_by_id(&connection, result_asset_id)? else {
            continue;
        };
        let Some(person_asset) = asset_summary_by_id(&connection, person_asset_id)? else {
            continue;
        };
        let Some(garment_asset) = asset_summary_by_id(&connection, garment_asset_id)? else {
            continue;
        };
        if !Path::new(&asset.stored_path).is_file() {
            continue;
        }
        if let Some(object) = result.as_object_mut() {
            object.insert("task_id".to_string(), serde_json::json!(task_id));
            object.insert("created_at".to_string(), serde_json::json!(created_at));
            object.insert("asset".to_string(), serde_json::json!(asset));
            object.insert("person_asset".to_string(), serde_json::json!(person_asset));
            object.insert("garment_asset".to_string(), serde_json::json!(garment_asset));
        }
        results.push(result);
    }
    Ok(results)
}

fn preflight_status(result: &serde_json::Value) -> &str {
    result
        .get("status")
        .and_then(|value| value.as_str())
        .unwrap_or("block")
}

#[tauri::command]
fn run_tryon_preflight(
    app: tauri::AppHandle,
    person_asset_id: String,
    garment_asset_id: String,
    garment_photo_type: String,
) -> Result<serde_json::Value, String> {
    if !matches!(garment_photo_type.as_str(), "model" | "flat-lay") {
        return Err("服装照片类型无效".to_string());
    }
    let connection = open_database(&app)?;
    let person = asset_summary_by_id(&connection, &person_asset_id)?
        .ok_or_else(|| "人物素材不存在或已删除".to_string())?;
    let garment = asset_summary_by_id(&connection, &garment_asset_id)?
        .ok_or_else(|| "服装素材不存在或已删除".to_string())?;
    let task_id = Uuid::new_v4().to_string();
    let created_at = Utc::now().to_rfc3339();
    let parameters = serde_json::json!({
        "person_asset_id": person_asset_id,
        "garment_asset_id": garment_asset_id,
        "category": "upper",
        "garment_photo_type": &garment_photo_type,
        "preflight_revision": "tryon-preflight-v2",
    });
    connection
        .execute(
            "INSERT INTO tasks(id, task_type, status, stage, parameters_json, created_at, started_at)
             VALUES(?1, 'tryon.preflight', 'running', 'image_checks', ?2, ?3, ?3)",
            params![task_id, parameters.to_string(), created_at],
        )
        .map_err(|error| error.to_string())?;
    drop(connection);

    let execution = (|| -> Result<serde_json::Value, String> {
        let (_, person_result) = call_engine(
            "image.preflight",
            serde_json::json!({ "path": person.stored_path, "role": "person" }),
        )?;
        let (_, garment_result) = call_engine(
            "image.preflight",
            serde_json::json!({ "path": garment.stored_path, "role": "garment" }),
        )?;
        let scene = call_tryon_engine(
            &app,
            "tryon.analyze",
            serde_json::json!({
                "person_path": &person.stored_path,
                "garment_path": &garment.stored_path,
                "human_parser_dir": tryon_human_parser_dir(),
                "garment_photo_type": &garment_photo_type,
            }),
        )?;
        let scene_supported = scene
            .get("supported")
            .and_then(|value| value.as_bool())
            .unwrap_or(false);
        let status = if !scene_supported
            || preflight_status(&person_result) == "block"
            || preflight_status(&garment_result) == "block"
        {
            "block"
        } else if preflight_status(&person_result) == "warning"
            || preflight_status(&garment_result) == "warning"
        {
            "warning"
        } else {
            "pass"
        };
        Ok(serde_json::json!({
            "task_id": task_id,
            "status": status,
            "person": person_result,
            "garment": garment_result,
            "scene": scene,
            "ran_at": Utc::now().to_rfc3339(),
        }))
    })();

    let connection = open_database(&app)?;
    match execution {
        Ok(result) => {
            let finished_at = Utc::now().to_rfc3339();
            connection
                .execute(
                    "UPDATE tasks SET status = 'succeeded', stage = 'complete', finished_at = ?2, result_json = ?3 WHERE id = ?1",
                    params![task_id, finished_at, result.to_string()],
                )
                .map_err(|error| error.to_string())?;
            for (asset_id, role) in [(&person_asset_id, "person"), (&garment_asset_id, "garment")] {
                let role_result = result.get(role).cloned().unwrap_or(serde_json::Value::Null);
                connection
                    .execute(
                        "INSERT INTO asset_analysis(asset_id, analysis_type, revision, status, result_json, created_at)
                         VALUES(?1, ?2, 'tryon-preflight-v2', ?3, ?4, ?5)
                         ON CONFLICT(asset_id, analysis_type, revision) DO UPDATE SET
                           status = excluded.status,
                           result_json = excluded.result_json,
                           created_at = excluded.created_at",
                        params![asset_id, format!("tryon.{role}"), preflight_status(&role_result), role_result.to_string(), finished_at],
                    )
                    .map_err(|error| error.to_string())?;
            }
            Ok(result)
        }
        Err(error) => {
            connection
                .execute(
                    "UPDATE tasks SET status = 'failed', stage = 'image_checks', finished_at = ?2, error_code = 'preflight_failed' WHERE id = ?1",
                    params![task_id, Utc::now().to_rfc3339()],
                )
                .map_err(|database_error| database_error.to_string())?;
            Err(error)
        }
    }
}

const PLANT_MODEL_REVISION: &str = "ce901ab3c6a913f9e9ef94ce6d27761069f4f01c";

fn plant_weights_dir(app: &tauri::AppHandle) -> Result<PathBuf, String> {
    let installed = app_data_dir(app)?
        .join("models")
        .join("plants")
        .join("weights");
    if installed.join("open_clip_pytorch_model.bin").is_file() {
        return Ok(installed);
    }
    Ok(project_root().join("models").join("plants").join("weights"))
}

fn call_plant_engine(
    operation: &str,
    payload: serde_json::Value,
) -> Result<serde_json::Value, String> {
    let root = project_root();
    let python = root
        .join("runtime")
        .join("plants")
        .join(".venv")
        .join("Scripts")
        .join("python.exe");
    if !python.is_file() {
        return Err("植物识别隔离运行时尚未安装".to_string());
    }
    let mut child = Command::new(python)
        .arg(root.join("engine").join("plant_worker.py"))
        .current_dir(&root)
        .env("HF_HUB_OFFLINE", "1")
        .env("TRANSFORMERS_OFFLINE", "1")
        .env("PYTHONIOENCODING", "utf-8")
        .env("PYTHONUTF8", "1")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| format!("无法启动植物识别运行时：{error}"))?;
    if let Some(mut stdin) = child.stdin.take() {
        let request = serde_json::json!({
            "protocol_version": 1,
            "request_id": Uuid::new_v4(),
            "op": operation,
            "payload": payload,
        });
        writeln!(stdin, "{request}").map_err(|_| "无法写入植物引擎请求".to_string())?;
    }
    let output = child
        .wait_with_output()
        .map_err(|error| format!("等待植物识别引擎失败：{error}"))?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let response_line = stdout
        .lines()
        .rev()
        .find(|line| !line.trim().is_empty())
        .ok_or_else(|| format!("植物引擎没有返回结果：{}", String::from_utf8_lossy(&output.stderr).trim()))?;
    let response = serde_json::from_str::<EngineEnvelope>(response_line)
        .map_err(|error| format!("无法解析植物引擎响应：{error}"))?;
    if !response.success {
        return Err(response
            .error
            .map(|error| format!("{}：{}", error.code, error.message))
            .unwrap_or_else(|| "植物引擎返回失败状态".to_string()));
    }
    response.payload.ok_or_else(|| "植物引擎响应缺少 payload".to_string())
}

#[tauri::command]
fn get_plant_model_status(app: tauri::AppHandle) -> Result<serde_json::Value, String> {
    let result = call_plant_engine(
        "plant.status",
        serde_json::json!({ "weights_dir": plant_weights_dir(&app)? }),
    )?;
    let ready = result.get("ready").and_then(|value| value.as_bool()).unwrap_or(false);
    let connection = open_database(&app)?;
    if ready {
        connection.execute(
            "INSERT INTO installed_models(id, revision, status, installed_at)
             VALUES('bioclip-vit-b-16', ?1, 'available', ?2)
             ON CONFLICT(id) DO UPDATE SET revision = excluded.revision, status = excluded.status",
            params![PLANT_MODEL_REVISION, Utc::now().to_rfc3339()],
        ).map_err(|error| error.to_string())?;
    } else {
        connection.execute("DELETE FROM installed_models WHERE id = 'bioclip-vit-b-16'", [])
            .map_err(|error| error.to_string())?;
    }
    Ok(result)
}

fn plant_observation_by_id(
    connection: &Connection,
    observation_id: &str,
) -> Result<Option<serde_json::Value>, String> {
    let stored = connection.query_row(
        "SELECT o.id, o.organ, o.note, o.status, o.confirmed_taxon_id,
                o.confirmed_chinese_name, o.prediction_json, o.model_revision,
                o.created_at, o.updated_at,
                a.id, a.original_name, a.stored_path, a.mime_type, a.size_bytes, a.created_at
         FROM plant_observations o JOIN assets a ON a.id = o.asset_id
         WHERE o.id = ?1 AND a.deleted_at IS NULL",
        [observation_id],
        |row| Ok((
            row.get::<_, String>(0)?, row.get::<_, String>(1)?, row.get::<_, String>(2)?,
            row.get::<_, String>(3)?, row.get::<_, Option<String>>(4)?, row.get::<_, Option<String>>(5)?,
            row.get::<_, String>(6)?, row.get::<_, String>(7)?, row.get::<_, String>(8)?,
            row.get::<_, String>(9)?, row.get::<_, String>(10)?, row.get::<_, String>(11)?,
            row.get::<_, String>(12)?, row.get::<_, String>(13)?, row.get::<_, u64>(14)?, row.get::<_, String>(15)?,
        )),
    ).optional().map_err(|error| error.to_string())?;
    let Some((id, organ, note, status, confirmed_taxon_id, confirmed_chinese_name,
        prediction_json, model_revision, created_at, updated_at, asset_id, original_name,
        stored_path, mime_type, size_bytes, asset_created_at)) = stored else { return Ok(None); };
    let prediction = serde_json::from_str::<serde_json::Value>(&prediction_json)
        .unwrap_or(serde_json::Value::Null);
    Ok(Some(serde_json::json!({
        "id": id, "organ": organ, "note": note, "status": status,
        "confirmed_taxon_id": confirmed_taxon_id,
        "confirmed_chinese_name": confirmed_chinese_name,
        "prediction": prediction, "model_revision": model_revision,
        "created_at": created_at, "updated_at": updated_at,
        "asset": { "id": asset_id, "original_name": original_name, "stored_path": stored_path,
            "mime_type": mime_type, "size_bytes": size_bytes, "created_at": asset_created_at }
    })))
}

#[tauri::command]
fn identify_plant(
    app: tauri::AppHandle,
    asset_id: String,
    organ: String,
    note: String,
) -> Result<serde_json::Value, String> {
    if !matches!(organ.as_str(), "whole" | "leaf" | "flower" | "fruit") {
        return Err("观察部位无效".to_string());
    }
    if note.chars().count() > 500 {
        return Err("观察备注不能超过 500 个字符".to_string());
    }
    let connection = open_database(&app)?;
    let asset = asset_summary_by_id(&connection, &asset_id)?
        .ok_or_else(|| "植物素材不存在或已删除".to_string())?;
    let task_id = Uuid::new_v4().to_string();
    let observation_id = Uuid::new_v4().to_string();
    let now = Utc::now().to_rfc3339();
    connection.execute(
        "INSERT INTO tasks(id, task_type, status, stage, parameters_json, created_at, started_at)
         VALUES(?1, 'plant.identify', 'running', 'local_retrieval', ?2, ?3, ?3)",
        params![task_id, serde_json::json!({"asset_id": asset_id, "organ": organ}).to_string(), now],
    ).map_err(|error| error.to_string())?;
    drop(connection);

    let execution = call_plant_engine("plant.identify", serde_json::json!({
        "image_path": asset.stored_path,
        "weights_dir": plant_weights_dir(&app)?,
        "organ": organ,
    }));
    let finished_at = Utc::now().to_rfc3339();
    let connection = open_database(&app)?;
    match execution {
        Ok(mut prediction) => {
            if let Some(object) = prediction.as_object_mut() {
                object.insert("task_id".to_string(), serde_json::json!(task_id));
                object.insert("observation_id".to_string(), serde_json::json!(observation_id));
            }
            connection.execute(
                "INSERT INTO plant_observations(id, asset_id, organ, note, status, prediction_json, model_revision, created_at, updated_at)
                 VALUES(?1, ?2, ?3, ?4, 'pending', ?5, ?6, ?7, ?7)",
                params![observation_id, asset_id, organ, note, prediction.to_string(), PLANT_MODEL_REVISION, finished_at],
            ).map_err(|error| error.to_string())?;
            connection.execute(
                "INSERT INTO asset_analysis(asset_id, analysis_type, revision, status, result_json, created_at)
                 VALUES(?1, 'plant.identify', ?2, 'uncertain', ?3, ?4)
                 ON CONFLICT(asset_id, analysis_type, revision) DO UPDATE SET result_json = excluded.result_json, created_at = excluded.created_at",
                params![asset_id, PLANT_MODEL_REVISION, prediction.to_string(), finished_at],
            ).map_err(|error| error.to_string())?;
            connection.execute(
                "UPDATE tasks SET status = 'succeeded', stage = 'complete', finished_at = ?2, result_json = ?3 WHERE id = ?1",
                params![task_id, finished_at, prediction.to_string()],
            ).map_err(|error| error.to_string())?;
            plant_observation_by_id(&connection, &observation_id)?.ok_or_else(|| "无法读取新观察记录".to_string())
        }
        Err(error) => {
            connection.execute(
                "UPDATE tasks SET status = 'failed', finished_at = ?2, error_code = 'plant_identify_failed' WHERE id = ?1",
                params![task_id, finished_at],
            ).map_err(|database_error| database_error.to_string())?;
            Err(error)
        }
    }
}

#[tauri::command]
fn list_plant_observations(app: tauri::AppHandle, limit: Option<u32>) -> Result<Vec<serde_json::Value>, String> {
    let connection = open_database(&app)?;
    let mut statement = connection.prepare(
        "SELECT id FROM plant_observations ORDER BY created_at DESC LIMIT ?1"
    ).map_err(|error| error.to_string())?;
    let ids = statement.query_map([limit.unwrap_or(40).min(100)], |row| row.get::<_, String>(0))
        .map_err(|error| error.to_string())?
        .collect::<Result<Vec<_>, _>>().map_err(|error| error.to_string())?;
    ids.iter().map(|id| plant_observation_by_id(&connection, id)?.ok_or_else(|| "观察记录素材不存在".to_string())).collect()
}

#[tauri::command]
fn confirm_plant_observation(
    app: tauri::AppHandle,
    observation_id: String,
    taxon_id: Option<String>,
) -> Result<serde_json::Value, String> {
    let connection = open_database(&app)?;
    let observation = plant_observation_by_id(&connection, &observation_id)?
        .ok_or_else(|| "观察记录不存在".to_string())?;
    let candidates = observation.pointer("/prediction/candidates").and_then(|value| value.as_array())
        .ok_or_else(|| "观察记录没有可确认候选".to_string())?;
    let confirmed_name = if let Some(ref requested) = taxon_id {
        Some(candidates.iter().find(|candidate| candidate.get("taxon_id").and_then(|value| value.as_str()) == Some(requested.as_str()))
            .and_then(|candidate| candidate.get("chinese_name")).and_then(|value| value.as_str())
            .ok_or_else(|| "只能确认本次结果中的候选".to_string())?.to_string())
    } else { None };
    let status = if taxon_id.is_some() { "confirmed" } else { "unresolved" };
    connection.execute(
        "UPDATE plant_observations SET status = ?2, confirmed_taxon_id = ?3,
         confirmed_chinese_name = ?4, updated_at = ?5 WHERE id = ?1",
        params![observation_id, status, taxon_id, confirmed_name, Utc::now().to_rfc3339()],
    ).map_err(|error| error.to_string())?;
    plant_observation_by_id(&connection, &observation_id)?.ok_or_else(|| "观察记录不存在".to_string())
}

fn call_lab_engine(payload: serde_json::Value) -> Result<serde_json::Value, String> {
    let root = project_root();
    let mut child = Command::new("python")
        .arg(root.join("engine").join("lab_worker.py"))
        .current_dir(&root)
        .env("PYTHONIOENCODING", "utf-8")
        .env("PYTHONUTF8", "1")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| format!("无法启动实验室图像引擎：{error}"))?;
    if let Some(mut stdin) = child.stdin.take() {
        let request = serde_json::json!({
            "protocol_version": 1, "request_id": Uuid::new_v4(),
            "op": "lab.segment", "payload": payload,
        });
        writeln!(stdin, "{request}").map_err(|_| "无法写入实验室引擎请求".to_string())?;
    }
    let output = child.wait_with_output().map_err(|error| format!("等待实验室图像引擎失败：{error}"))?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let response_line = stdout.lines().rev().find(|line| !line.trim().is_empty())
        .ok_or_else(|| format!("实验室引擎没有返回结果：{}", String::from_utf8_lossy(&output.stderr).trim()))?;
    let response = serde_json::from_str::<EngineEnvelope>(response_line)
        .map_err(|error| format!("无法解析实验室引擎响应：{error}"))?;
    if !response.success {
        return Err(response.error.map(|error| format!("{}：{}", error.code, error.message))
            .unwrap_or_else(|| "实验室引擎返回失败状态".to_string()));
    }
    response.payload.ok_or_else(|| "实验室引擎响应缺少 payload".to_string())
}

fn lab_analysis_by_id(connection: &Connection, analysis_id: &str) -> Result<Option<serde_json::Value>, String> {
    let stored = connection.query_row(
        "SELECT l.id, l.result_json, l.overlay_path, l.review_status, l.created_at, l.updated_at,
                a.id, a.original_name, a.stored_path, a.mime_type, a.size_bytes, a.created_at
         FROM lab_analyses l JOIN assets a ON a.id = l.asset_id
         WHERE l.id = ?1 AND a.deleted_at IS NULL",
        [analysis_id],
        |row| Ok((
            row.get::<_, String>(0)?, row.get::<_, String>(1)?, row.get::<_, String>(2)?,
            row.get::<_, String>(3)?, row.get::<_, String>(4)?, row.get::<_, String>(5)?,
            row.get::<_, String>(6)?, row.get::<_, String>(7)?, row.get::<_, String>(8)?,
            row.get::<_, String>(9)?, row.get::<_, u64>(10)?, row.get::<_, String>(11)?,
        )),
    ).optional().map_err(|error| error.to_string())?;
    let Some((id, result_json, overlay_path, review_status, created_at, updated_at,
        asset_id, original_name, stored_path, mime_type, size_bytes, asset_created_at)) = stored else { return Ok(None); };
    let result = serde_json::from_str::<serde_json::Value>(&result_json).unwrap_or(serde_json::Value::Null);
    Ok(Some(serde_json::json!({
        "id": id, "result": result, "overlay_path": overlay_path,
        "review_status": review_status, "created_at": created_at, "updated_at": updated_at,
        "asset": { "id": asset_id, "original_name": original_name, "stored_path": stored_path,
            "mime_type": mime_type, "size_bytes": size_bytes, "created_at": asset_created_at }
    })))
}

#[tauri::command]
fn run_lab_segmentation(
    app: tauri::AppHandle,
    asset_id: String,
    threshold_scale: f64,
    min_area: u32,
    exclude_edges: bool,
    scale_um_per_px: Option<f64>,
) -> Result<serde_json::Value, String> {
    if !(0.5..=1.5).contains(&threshold_scale) { return Err("阈值系数必须在 0.5 到 1.5 之间".to_string()); }
    if !(4..=100_000).contains(&min_area) { return Err("最小核面积必须在 4 到 100000 像素之间".to_string()); }
    if scale_um_per_px.is_some_and(|value| value <= 0.0 || value > 10_000.0) { return Err("像素尺度无效".to_string()); }
    let connection = open_database(&app)?;
    let asset = asset_summary_by_id(&connection, &asset_id)?.ok_or_else(|| "实验图像素材不存在或已删除".to_string())?;
    let task_id = Uuid::new_v4().to_string();
    let analysis_id = Uuid::new_v4().to_string();
    let now = Utc::now().to_rfc3339();
    connection.execute(
        "INSERT INTO tasks(id, task_type, status, stage, parameters_json, created_at, started_at)
         VALUES(?1, 'lab.segment', 'running', 'segmentation', ?2, ?3, ?3)",
        params![task_id, serde_json::json!({"asset_id": asset_id, "threshold_scale": threshold_scale, "min_area": min_area, "exclude_edges": exclude_edges, "scale_um_per_px": scale_um_per_px}).to_string(), now],
    ).map_err(|error| error.to_string())?;
    drop(connection);
    let lab_dir = app_data_dir(&app)?.join("lab");
    fs::create_dir_all(&lab_dir).map_err(|error| error.to_string())?;
    let overlay_path = lab_dir.join(format!("{analysis_id}.overlay.png"));
    let execution = call_lab_engine(serde_json::json!({
        "image_path": asset.stored_path, "output_path": overlay_path,
        "threshold_scale": threshold_scale, "min_area": min_area,
        "exclude_edges": exclude_edges, "scale_um_per_px": scale_um_per_px,
    }));
    let finished_at = Utc::now().to_rfc3339();
    let connection = open_database(&app)?;
    match execution {
        Ok(mut result) => {
            if let Some(object) = result.as_object_mut() {
                object.insert("task_id".to_string(), serde_json::json!(task_id));
                object.insert("analysis_id".to_string(), serde_json::json!(analysis_id));
            }
            connection.execute(
                "INSERT INTO lab_analyses(id, asset_id, result_json, overlay_path, review_status, created_at, updated_at)
                 VALUES(?1, ?2, ?3, ?4, 'unreviewed', ?5, ?5)",
                params![analysis_id, asset_id, result.to_string(), overlay_path.to_string_lossy(), finished_at],
            ).map_err(|error| error.to_string())?;
            connection.execute(
                "UPDATE tasks SET status = 'succeeded', stage = 'complete', finished_at = ?2, result_json = ?3 WHERE id = ?1",
                params![task_id, finished_at, result.to_string()],
            ).map_err(|error| error.to_string())?;
            lab_analysis_by_id(&connection, &analysis_id)?.ok_or_else(|| "无法读取新分析".to_string())
        }
        Err(error) => {
            let _ = fs::remove_file(&overlay_path);
            connection.execute(
                "UPDATE tasks SET status = 'failed', finished_at = ?2, error_code = 'lab_segment_failed' WHERE id = ?1",
                params![task_id, finished_at],
            ).map_err(|database_error| database_error.to_string())?;
            Err(error)
        }
    }
}

#[tauri::command]
fn list_lab_analyses(app: tauri::AppHandle, limit: Option<u32>) -> Result<Vec<serde_json::Value>, String> {
    let connection = open_database(&app)?;
    let mut statement = connection.prepare("SELECT id FROM lab_analyses ORDER BY created_at DESC LIMIT ?1")
        .map_err(|error| error.to_string())?;
    let ids = statement.query_map([limit.unwrap_or(30).min(100)], |row| row.get::<_, String>(0))
        .map_err(|error| error.to_string())?.collect::<Result<Vec<_>, _>>().map_err(|error| error.to_string())?;
    ids.iter().map(|id| lab_analysis_by_id(&connection, id)?.ok_or_else(|| "分析素材不存在".to_string())).collect()
}

#[tauri::command]
fn review_lab_analysis(app: tauri::AppHandle, analysis_id: String) -> Result<serde_json::Value, String> {
    let connection = open_database(&app)?;
    let changed = connection.execute(
        "UPDATE lab_analyses SET review_status = 'reviewed', updated_at = ?2 WHERE id = ?1",
        params![analysis_id, Utc::now().to_rfc3339()],
    ).map_err(|error| error.to_string())?;
    if changed == 0 { return Err("分析记录不存在".to_string()); }
    lab_analysis_by_id(&connection, &analysis_id)?.ok_or_else(|| "分析记录不存在".to_string())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .setup(|app| {
            recover_interrupted_tasks(app.handle()).map_err(std::io::Error::other)?;
            Ok(())
        })
        .plugin(tauri_plugin_dialog::init())
        .invoke_handler(tauri::generate_handler![
            initialize_app,
            complete_onboarding,
            import_asset,
            list_recent_assets,
            add_photo_source,
            list_photo_sources,
            remove_photo_source,
            start_photo_scan,
            cancel_photo_scan,
            list_uncategorized_photos,
            list_classified_photos,
            set_photo_category,
            set_photo_categories,
            import_device_photo,
            get_category_preferences,
            save_category_preferences,
            get_album_classifier_status,
            start_album_classification,
            list_tasks,
            cancel_task,
            load_tryon_draft,
            save_tryon_draft,
            check_engine_health,
            get_tryon_model_status,
            get_hand_refiner_status,
            run_tryon_preflight,
            run_tryon_generation,
            list_tryon_results,
            get_plant_model_status,
            identify_plant,
            list_plant_observations,
            confirm_plant_observation,
            run_lab_segmentation,
            list_lab_analyses,
            review_lab_analysis,
        ])
        .run(tauri::generate_context!())
        .expect("error while running Vision Analytics Studio");
}
