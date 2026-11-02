import { useEffect, useMemo, useState } from "react";
import {
  Aperture,
  Beaker,
  Bell,
  Check,
  ChevronRight,
  CircleAlert,
  CloudOff,
  Cpu,
  FolderOpen,
  FolderPlus,
  Gauge,
  HardDrive,
  Home,
  Image as ImageIcon,
  ImagePlus,
  Layers3,
  Leaf,
  LoaderCircle,
  LogIn,
  Palette,
  RefreshCw,
  Search,
  Settings,
  ShieldCheck,
  Shirt,
  Sparkles,
  Trash2,
  Upload,
  UserRound,
  WandSparkles,
  X,
} from "lucide-react";
import {
  assetUrl,
  cancelPhotoScan,
  cancelTask,
  checkEngine,
  chooseAndImportAsset,
  chooseAndImportAssets,
  choosePhotoSource,
  devicePhotoThumbnailUrl,
  completeOnboarding,
  initializeApp,
  getCategoryPreferences,
  getAlbumClassifierStatus,
  getHandRefinerStatus,
  getPlantModelStatus,
  getTryOnModelStatus,
  importDevicePhoto,
  identifyPlant,
  labOverlayUrl,
  listClassifiedPhotos,
  listPhotoSources,
  listPlantObservations,
  listLabAnalyses,
  listRecentAssets,
  listTasks,
  listTryOnResults,
  listUncategorizedPhotos,
  loadTryOnDraft,
  runTryOnPreflight,
  runTryOnGeneration,
  removePhotoSource,
  confirmPlantObservation,
  reviewLabAnalysis,
  runLabSegmentation,
  saveCategoryPreferences,
  saveTryOnDraft,
  setPhotoCategory,
  setPhotoCategories,
  startAlbumClassification,
  startPhotoScan,
  tryOnResultUrl,
} from "./lib/desktop";
import { loginAccount, logoutAccount, registerAccount } from "./lib/cloud";
import type {
  AssetSummary,
  AlbumClassifierStatus,
  CategoryPreference,
  CloudSession,
  DashboardSnapshot,
  DevicePhotoSummary,
  EngineHealth,
  HandRefinerStatus,
  ImagePreflightResult,
  PhotoSourceSummary,
  PlantModelStatus,
  PlantObservation,
  LabAnalysis,
  TaskSummary,
  TryOnPreflightResult,
  TryOnHistoryItem,
  TryOnModelStatus,
  WorkspaceId,
} from "./types";

const modules = [
  {
    id: "tryon" as const,
    title: "虚拟穿搭",
    eyebrow: "TRY-ON STUDIO",
    description: "导入人物与服装照片，在本机生成穿搭预览。",
    icon: Shirt,
    className: "module-card--violet",
    status: "技术验证中",
    action: "进入工作台",
  },
  {
    id: "plants" as const,
    title: "植物与农业",
    eyebrow: "PLANT LAB",
    description: "识别有限植物集，记录可见状态与观察历史。",
    icon: Leaf,
    className: "module-card--green",
    status: "本地候选识别",
    action: "新建观察",
  },
  {
    id: "laboratory" as const,
    title: "实验室图像",
    eyebrow: "MICRO ANALYSIS",
    description: "分割二维核图，复核数量、尺度与形态统计。",
    icon: Beaker,
    className: "module-card--blue",
    status: "CPU 分割基线",
    action: "开始分析",
  },
  {
    id: "album" as const,
    title: "个人相册",
    eyebrow: "LOCAL LIBRARY",
    description: "离线整理照片、标签、搜索与跨模块素材。",
    icon: ImageIcon,
    className: "module-card--amber",
    status: "基础能力",
    action: "打开相册",
  },
];

const navItems: { id: WorkspaceId; label: string; icon: typeof Home }[] = [
  { id: "home", label: "首页", icon: Home },
  { id: "tryon", label: "虚拟穿搭", icon: Shirt },
  { id: "plants", label: "植物与农业", icon: Leaf },
  { id: "laboratory", label: "实验室图像", icon: Beaker },
  { id: "album", label: "个人相册", icon: ImageIcon },
];

const utilityItems: { id: WorkspaceId; label: string; icon: typeof Home }[] = [
  { id: "tasks", label: "任务中心", icon: Layers3 },
  { id: "models", label: "模型中心", icon: Cpu },
  { id: "settings", label: "设置", icon: Settings },
];

const emptySnapshot: DashboardSnapshot = {
  onboarding_complete: false,
  app_data_dir: "正在准备…",
  asset_count: 0,
  active_task_count: 0,
  installed_model_count: 0,
  device: { gpu_name: null, vram_mib: null, gpu_available: false },
};

function formatBytes(value: number) {
  if (value < 1024) return `${value} B`;
  if (value < 1024 ** 2) return `${(value / 1024).toFixed(1)} KB`;
  if (value >= 1024 ** 3) return `${(value / 1024 ** 3).toFixed(2)} GB`;
  return `${(value / 1024 ** 2).toFixed(1)} MB`;
}

function App() {
  const [active, setActive] = useState<WorkspaceId>("home");
  const [snapshot, setSnapshot] = useState<DashboardSnapshot>(emptySnapshot);
  const [recentAssets, setRecentAssets] = useState<AssetSummary[]>([]);
  const [tasks, setTasks] = useState<TaskSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [importing, setImporting] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  const [accountOpen, setAccountOpen] = useState(false);
  const [engine, setEngine] = useState<EngineHealth | null>(null);
  const [engineChecking, setEngineChecking] = useState(false);
  const [tryOnModel, setTryOnModel] = useState<TryOnModelStatus | null>(null);
  const [handRefiner, setHandRefiner] = useState<HandRefinerStatus | null>(null);
  const [plantModel, setPlantModel] = useState<PlantModelStatus | null>(null);
  const [cloudSession, setCloudSession] = useState<CloudSession | null>(null);

  const refresh = async () => {
    const [nextSnapshot, assets, nextTasks] = await Promise.all([initializeApp(), listRecentAssets(), listTasks()]);
    setSnapshot(nextSnapshot);
    setRecentAssets(assets);
    setTasks(nextTasks);
  };

  useEffect(() => {
    refresh()
      .catch((error) => setNotice(String(error)))
      .finally(() => setLoading(false));
    getTryOnModelStatus().then(setTryOnModel).catch(() => setTryOnModel(null));
    getHandRefinerStatus().then(setHandRefiner).catch(() => setHandRefiner(null));
    getPlantModelStatus().then(setPlantModel).catch(() => setPlantModel(null));
  }, []);

  const activeLabel = useMemo(
    () => [...navItems, ...utilityItems].find((item) => item.id === active)?.label ?? "首页",
    [active],
  );

  const finishOnboarding = async () => {
    setLoading(true);
    try {
      setSnapshot(await completeOnboarding());
    } catch (error) {
      setNotice(String(error));
    } finally {
      setLoading(false);
    }
  };

  const importImages = async () => {
    setImporting(true);
    setNotice(null);
    try {
      const imported = await chooseAndImportAssets();
      if (imported.length) {
        await refresh();
        setNotice(`已将 ${imported.length} 张图片复制到本地素材库`);
      }
    } catch (error) {
      setNotice(error instanceof Error ? error.message : String(error));
    } finally {
      setImporting(false);
    }
  };

  const runEngineCheck = async () => {
    setEngineChecking(true);
    try {
      const [nextEngine, nextTryOnModel, nextHandRefiner, nextPlantModel] = await Promise.all([checkEngine(), getTryOnModelStatus(), getHandRefinerStatus(), getPlantModelStatus()]);
      setEngine(nextEngine);
      setTryOnModel(nextTryOnModel);
      setHandRefiner(nextHandRefiner);
      setPlantModel(nextPlantModel);
      await refresh();
    } catch (error) {
      setNotice(error instanceof Error ? error.message : String(error));
    } finally {
      setEngineChecking(false);
    }
  };

  if (loading) {
    return (
      <div className="boot-screen">
        <div className="brand-mark"><Aperture size={28} /></div>
        <LoaderCircle className="spin" size={24} />
        <span>正在准备设备缓存</span>
      </div>
    );
  }

  return (
    <div className="app-shell">
      {!snapshot.onboarding_complete && <Onboarding onComplete={finishOnboarding} />}
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark"><Aperture size={22} /></div>
          <div><strong>Vision</strong><span>Analytics Studio</span></div>
        </div>

        <nav className="nav-group" aria-label="工作区">
          <span className="nav-caption">工作空间</span>
          {navItems.map(({ id, label, icon: Icon }) => (
            <button key={id} className={active === id ? "nav-item is-active" : "nav-item"} onClick={() => setActive(id)}>
              <Icon size={18} strokeWidth={1.8} /><span>{label}</span>
            </button>
          ))}
        </nav>

        <nav className="nav-group nav-group--utility" aria-label="应用">
          <span className="nav-caption">应用</span>
          {utilityItems.map(({ id, label, icon: Icon }) => (
            <button key={id} className={active === id ? "nav-item is-active" : "nav-item"} onClick={() => setActive(id)}>
              <Icon size={18} strokeWidth={1.8} /><span>{label}</span>
              {id === "tasks" && snapshot.active_task_count > 0 && <b>{snapshot.active_task_count}</b>}
            </button>
          ))}
        </nav>

        <button className="local-profile" onClick={() => setAccountOpen(true)}>
          <span className="avatar">{cloudSession?.display_name.slice(0, 1) ?? "开"}</span>
          <span><strong>{cloudSession?.display_name ?? "开发模式"}</strong><small>{cloudSession ? <><ShieldCheck size={12} /> 已连接账户服务</> : <><CloudOff size={12} /> 尚未登录</>}</small></span>
          <ChevronRight size={16} />
        </button>
      </aside>

      <main className="main-area">
        <header className="topbar">
          <div><span className="crumb">工作空间</span><strong>{activeLabel}</strong></div>
          <div className="topbar-actions">
            <label className="search-box"><Search size={17} /><input aria-label="搜索" placeholder="搜索素材、任务或功能" /></label>
            <button className="icon-button" aria-label="通知"><Bell size={18} /></button>
            <button className="account-button" onClick={() => setAccountOpen(true)}><UserRound size={17} />{cloudSession?.display_name ?? "登录 / 注册"}</button>
          </div>
        </header>

        {notice && <div className="notice"><Check size={17} /><span>{notice}</span><button onClick={() => setNotice(null)}><X size={16} /></button></div>}

        <div className="content">
          {active === "home" ? (
            <Dashboard
              snapshot={snapshot}
              assets={recentAssets}
              importing={importing}
              onImport={importImages}
              onNavigate={setActive}
            />
          ) : active === "tryon" ? (
            <TryOnWorkspace
              modelStatus={tryOnModel}
              onLibraryChanged={refresh}
              onNavigate={setActive}
            />
          ) : active === "plants" ? (
            <PlantWorkspace modelStatus={plantModel} onLibraryChanged={refresh} onNavigate={setActive} />
          ) : active === "laboratory" ? (
            <LaboratoryWorkspace onLibraryChanged={refresh} />
          ) : active === "album" ? (
            <AlbumWorkspace assets={recentAssets} importing={importing} onImport={importImages} onNotice={setNotice} onNavigate={setActive} onLibraryChanged={refresh} />
          ) : active === "models" ? (
            <ModelCenter engine={engine} checking={engineChecking} onCheck={runEngineCheck} snapshot={snapshot} tryOnModel={tryOnModel} handRefiner={handRefiner} plantModel={plantModel} />
          ) : active === "tasks" ? (
            <TaskCenter tasks={tasks} onNavigate={setActive} onRefresh={refresh} onNotice={setNotice} />
          ) : (
            <WorkspacePlaceholder workspace={active} onImport={importImages} />
          )}
        </div>
      </main>

      {accountOpen && <AccountDialog session={cloudSession} onAuthenticated={setCloudSession} onClose={() => setAccountOpen(false)} />}
    </div>
  );
}

function Dashboard({ snapshot, assets, importing, onImport, onNavigate }: {
  snapshot: DashboardSnapshot;
  assets: AssetSummary[];
  importing: boolean;
  onImport: () => void;
  onNavigate: (id: WorkspaceId) => void;
}) {
  const gpuLabel = snapshot.device.gpu_available
    ? `${snapshot.device.gpu_name ?? "NVIDIA GPU"}${snapshot.device.vram_mib ? ` · ${(snapshot.device.vram_mib / 1024).toFixed(0)}GB` : ""}`
    : "未检测到可用 NVIDIA GPU";

  return (
    <>
      <section className="hero">
        <div>
          <span className="hero-kicker"><Sparkles size={15} /> 本地视觉工作台</span>
          <h1>让每一张图片，<br /><em>变成可用的结果。</em></h1>
          <p>从穿搭预览到科学测量，在同一个桌面应用中管理素材、任务与结果。</p>
          <div className="hero-actions">
            <button className="primary-button" onClick={onImport} disabled={importing}>
              {importing ? <LoaderCircle className="spin" size={18} /> : <Upload size={18} />}导入图片
            </button>
            <button className="secondary-button" onClick={() => onNavigate("tryon")}><Shirt size={18} />开始虚拟穿搭</button>
          </div>
        </div>
        <div className="hero-art" aria-hidden="true">
          <span className="orb orb--one" /><span className="orb orb--two" />
          <div className="art-card art-card--back"><Leaf size={30} /><span>LOCAL VISION</span></div>
          <div className="art-card art-card--front"><Aperture size={44} /><b>01</b><span>PRIVATE BY DEFAULT</span></div>
        </div>
      </section>

      <div className="section-heading"><div><span>四个工作区</span><h2>选择你要完成的任务</h2></div><small>功能将按验证结果逐步开放</small></div>
      <section className="module-grid">
        {modules.map(({ id, title, eyebrow, description, icon: Icon, className, status, action }) => (
          <button key={id} className={`module-card ${className}`} onClick={() => onNavigate(id)}>
            <div className="module-icon"><Icon size={25} /></div>
            <span className="module-eyebrow">{eyebrow}</span>
            <h3>{title}</h3><p>{description}</p>
            <div className="module-footer"><span>{status}</span><strong>{action}<ChevronRight size={15} /></strong></div>
          </button>
        ))}
      </section>

      <section className="lower-grid">
        <div className="panel recent-panel">
          <div className="panel-heading"><div><span>最近素材</span><h3>继续上次的工作</h3></div><button onClick={() => onNavigate("album")}>查看全部</button></div>
          {assets.length ? <div className="asset-list">{assets.map((asset) => (
            <div className="asset-row" key={asset.id}>
              <div className="file-icon"><ImageIcon size={20} /></div>
              <div><strong>{asset.original_name}</strong><span>{asset.mime_type} · {formatBytes(asset.size_bytes)}</span></div>
              <time>{new Date(asset.created_at).toLocaleDateString("zh-CN")}</time>
            </div>
          ))}</div> : <div className="empty-inline"><FolderOpen size={30} /><div><strong>还没有素材</strong><span>导入图片后会出现在这里</span></div><button onClick={onImport}>选择图片</button></div>}
        </div>
        <div className="panel status-panel">
          <div className="panel-heading"><div><span>运行状态</span><h3>这台电脑</h3></div><span className="status-dot">可用</span></div>
          <div className="device-row"><div className="device-icon"><Gauge size={21} /></div><div><strong>{gpuLabel}</strong><span>GPU 任务将按顺序执行</span></div></div>
          <div className="metric-grid">
            <div><b>{snapshot.asset_count}</b><span>本地素材</span></div>
            <div><b>{snapshot.active_task_count}</b><span>活跃任务</span></div>
            <div><b>{snapshot.installed_model_count}</b><span>已装模型</span></div>
          </div>
          <button className="text-button" onClick={() => onNavigate("models")}>检查本地引擎<ChevronRight size={15} /></button>
        </div>
      </section>
    </>
  );
}

function AlbumWorkspace({ assets, importing, onImport, onNotice, onNavigate, onLibraryChanged }: {
  assets: AssetSummary[];
  importing: boolean;
  onImport: () => void;
  onNotice: (message: string | null) => void;
  onNavigate: (id: WorkspaceId) => void;
  onLibraryChanged: () => Promise<void>;
}) {
  const [tab, setTab] = useState<"library" | "classified" | "categories" | "uncategorized">("library");
  const [sources, setSources] = useState<PhotoSourceSummary[]>([]);
  const [categories, setCategories] = useState<CategoryPreference[]>([]);
  const [uncategorized, setUncategorized] = useState<DevicePhotoSummary[]>([]);
  const [classified, setClassified] = useState<DevicePhotoSummary[]>([]);
  const [busySource, setBusySource] = useState<string | null>(null);
  const [savingCategories, setSavingCategories] = useState(false);
  const [assigningPhoto, setAssigningPhoto] = useState<string | null>(null);
  const [usingPhoto, setUsingPhoto] = useState<string | null>(null);
  const [classifier, setClassifier] = useState<AlbumClassifierStatus | null>(null);
  const [classificationTaskId, setClassificationTaskId] = useState<string | null>(null);
  const [selectedPhotos, setSelectedPhotos] = useState<Set<string>>(() => new Set());
  const [bulkCategory, setBulkCategory] = useState("");
  const [bulkSaving, setBulkSaving] = useState(false);

  const reloadAlbum = async () => {
    const [nextSources, nextCategories, nextUncategorized, nextClassified, classifierStatus] = await Promise.all([
      listPhotoSources(), getCategoryPreferences(), listUncategorizedPhotos(), listClassifiedPhotos(), getAlbumClassifierStatus(),
    ]);
    setSources(nextSources);
    setCategories(nextCategories);
    setUncategorized(nextUncategorized);
    setClassified(nextClassified);
    setClassifier(classifierStatus);
    setSelectedPhotos((current) => new Set([...current].filter((id) => nextUncategorized.some((photo) => photo.id === id))));
  };

  useEffect(() => {
    reloadAlbum().catch((error) => onNotice(error instanceof Error ? error.message : String(error)));
  }, []);

  const hasActiveScan = sources.some((source) => ["queued", "scanning", "thumbnailing"].includes(source.status));
  useEffect(() => {
    if (!hasActiveScan) {
      setBusySource(null);
      return;
    }
    const timer = window.setInterval(() => {
      reloadAlbum().catch((error) => onNotice(error instanceof Error ? error.message : String(error)));
    }, 900);
    return () => window.clearInterval(timer);
  }, [hasActiveScan]);

  useEffect(() => {
    if (!classificationTaskId) return;
    const timer = window.setInterval(async () => {
      try {
        const task = (await listTasks()).find((item) => item.id === classificationTaskId);
        await reloadAlbum();
        if (task && !["queued", "running"].includes(task.status)) {
          setClassificationTaskId(null);
          await onLibraryChanged();
          onNotice(task.status === "succeeded" ? "本地自动分类已完成；无法可靠判断的图片仍保留在未分类图库" : "本地自动分类未完成，请在任务中心查看原因");
        }
      } catch (error) {
        setClassificationTaskId(null);
        onNotice(error instanceof Error ? error.message : String(error));
      }
    }, 1000);
    return () => window.clearInterval(timer);
  }, [classificationTaskId]);

  const addSource = async () => {
    try {
      const source = await choosePhotoSource();
      if (!source) return;
      setBusySource(source.id);
      await startPhotoScan(source.id);
      await reloadAlbum();
      onNotice(`“${source.label}”已进入后台扫描，原图不会被复制或上传`);
    } catch (error) {
      setBusySource(null);
      onNotice(error instanceof Error ? error.message : String(error));
    }
  };

  const rescan = async (source: PhotoSourceSummary) => {
    setBusySource(source.id);
    try {
      await startPhotoScan(source.id);
      await reloadAlbum();
      onNotice(`正在后台更新“${source.label}”，扫描期间可以继续使用其他页面`);
    } catch (error) {
      setBusySource(null);
      onNotice(error instanceof Error ? error.message : String(error));
    }
  };

  const removeSource = async (source: PhotoSourceSummary) => {
    if (!window.confirm(`停止访问“${source.label}”并清除它的本机索引？原始照片不会被删除。`)) return;
    try {
      await removePhotoSource(source.id);
      await reloadAlbum();
      onNotice(`已移除“${source.label}”的目录授权和本机索引，原始照片未改动`);
    } catch (error) {
      onNotice(error instanceof Error ? error.message : String(error));
    }
  };

  const cancelScan = async (source: PhotoSourceSummary) => {
    try {
      await cancelPhotoScan(source.id);
      onNotice(`已请求停止“${source.label}”的扫描，正在安全结束当前步骤`);
      await reloadAlbum();
    } catch (error) {
      onNotice(error instanceof Error ? error.message : String(error));
    }
  };

  const saveCategories = async () => {
    setSavingCategories(true);
    try {
      setCategories(await saveCategoryPreferences(categories.filter((item) => item.enabled).map((item) => item.id)));
      onNotice("分类选择已保存在本机；本地模型只会使用这些类别");
    } catch (error) {
      onNotice(error instanceof Error ? error.message : String(error));
    } finally {
      setSavingCategories(false);
    }
  };

  const runAutoClassification = async () => {
    try {
      const taskId = await startAlbumClassification();
      setClassificationTaskId(taskId);
      onNotice("本地分类任务已开始，图片和识别结果不会上传");
      await onLibraryChanged();
    } catch (error) {
      onNotice(error instanceof Error ? error.message : String(error));
    }
  };

  const toggleSelectedPhoto = (photoId: string) => {
    setSelectedPhotos((current) => {
      const next = new Set(current);
      if (next.has(photoId)) next.delete(photoId);
      else next.add(photoId);
      return next;
    });
  };

  const applyBulkCategory = async () => {
    if (!selectedPhotos.size || !bulkCategory) return;
    setBulkSaving(true);
    try {
      const affected = await setPhotoCategories([...selectedPhotos], bulkCategory);
      const label = categories.find((item) => item.id === bulkCategory)?.label ?? bulkCategory;
      setSelectedPhotos(new Set());
      setBulkCategory("");
      await reloadAlbum();
      onNotice(`已将 ${affected} 张图片批量归入“${label}”`);
    } catch (error) {
      onNotice(error instanceof Error ? error.message : String(error));
    } finally {
      setBulkSaving(false);
    }
  };

  const assignCategory = async (photo: DevicePhotoSummary, categoryId: string | null) => {
    setAssigningPhoto(photo.id);
    try {
      await setPhotoCategory(photo.id, categoryId);
      setUncategorized((current) => current.filter((item) => item.id !== photo.id));
      if (categoryId) {
        const label = categories.find((item) => item.id === categoryId)?.label ?? categoryId;
        onNotice(`已将“${photo.file_name}”归入“${label}”`);
      } else {
        onNotice(`已取消“${photo.file_name}”的分类，它会回到未分类图库`);
      }
      await reloadAlbum();
    } catch (error) {
      onNotice(error instanceof Error ? error.message : String(error));
    } finally {
      setAssigningPhoto(null);
    }
  };

  const sourceStatus = (source: PhotoSourceSummary) => {
    if (source.status === "queued") return "等待扫描";
    if (source.status === "scanning") return "正在建立索引";
    if (source.status === "thumbnailing") return "正在生成缩略图";
    if (source.status === "failed") return "扫描失败，可重试";
    if (source.status === "interrupted") return "上次扫描已中断";
    return source.last_scanned_at ? `上次扫描 ${new Date(source.last_scanned_at).toLocaleString("zh-CN")}` : "尚未扫描";
  };

  const usePhotoForTryOn = async (photo: DevicePhotoSummary, role: "person" | "garment") => {
    setUsingPhoto(`${photo.id}:${role}`);
    try {
      const asset = await importDevicePhoto(photo.id);
      const draft = await loadTryOnDraft();
      await saveTryOnDraft(
        role === "person" ? asset.id : draft.person?.id ?? null,
        role === "garment" ? asset.id : draft.garment?.id ?? null,
        draft.category,
        draft.garment_photo_type,
        draft.sleeve_mode,
      );
      await onLibraryChanged();
      onNotice(`已把“${photo.file_name}”设置为穿搭${role === "person" ? "人物图" : "服装图"}`);
      onNavigate("tryon");
    } catch (error) {
      onNotice(error instanceof Error ? error.message : String(error));
    } finally {
      setUsingPhoto(null);
    }
  };

  const tryOnActions = (photo: DevicePhotoSummary) => <div className="photo-tryon-actions">
    <button disabled={usingPhoto !== null} onClick={() => usePhotoForTryOn(photo, "person")}>{usingPhoto === `${photo.id}:person` ? <LoaderCircle className="spin" size={13} /> : <UserRound size={13} />}人物图</button>
    <button disabled={usingPhoto !== null} onClick={() => usePhotoForTryOn(photo, "garment")}>{usingPhoto === `${photo.id}:garment` ? <LoaderCircle className="spin" size={13} /> : <Shirt size={13} />}服装图</button>
  </div>;

  return <div className="workspace-view">
    <div className="workspace-heading"><div><span>PRIVATE PHOTO LIBRARY</span><h1>个人相册</h1><p>仅索引你授权的目录，分类默认在本机完成。</p></div><button className="primary-button" onClick={addSource} disabled={busySource !== null}><FolderPlus size={18} />添加设备照片目录</button></div>
    <div className="album-tabs" role="tablist" aria-label="相册页面">
      <button className={tab === "library" ? "is-active" : ""} onClick={() => setTab("library")}>照片库</button>
      <button className={tab === "classified" ? "is-active" : ""} onClick={() => setTab("classified")}>分类图库 <b>{classified.length}</b></button>
      <button className={tab === "categories" ? "is-active" : ""} onClick={() => setTab("categories")}>分类设置</button>
      <button className={tab === "uncategorized" ? "is-active" : ""} onClick={() => setTab("uncategorized")}>未分类 <b>{uncategorized.length}</b></button>
    </div>

    {tab === "library" && <>
      <section className="album-source-panel">
        <div className="panel-heading"><div><span>设备照片来源</span><h3>你授权的文件夹</h3></div><small>递归读取图片元数据，不扫描文件夹外内容</small></div>
        {sources.length ? <div className="source-list">{sources.map((source) => <article key={source.id}>
          <div className="source-icon"><HardDrive size={20} /></div>
          <div><strong>{source.label}</strong><span title={source.path}>{source.path}</span><small>{source.photo_count} 张 · {sourceStatus(source)}</small></div>
          <div className="source-actions">{["queued", "scanning", "thumbnailing"].includes(source.status) ? <button className="secondary-button" onClick={() => cancelScan(source)}><X size={16} />停止扫描</button> : <button className="secondary-button" onClick={() => rescan(source)} disabled={busySource !== null}><RefreshCw size={16} />重新扫描</button>}<button className="icon-button" aria-label={`移除 ${source.label}`} onClick={() => removeSource(source)} disabled={busySource !== null}><Trash2 size={15} /></button></div>
        </article>)}</div> : <div className="source-empty"><FolderOpen size={28} /><div><strong>尚未授权设备照片目录</strong><span>选择“图片”或其他文件夹后，应用才会读取其中的照片。</span></div><button className="secondary-button" onClick={addSource}>选择文件夹</button></div>}
      </section>
      <div className="section-heading"><div><span>应用素材</span><h2>主动导入的图片副本</h2></div><button className="text-button" onClick={onImport} disabled={importing}><Upload size={16} />导入到素材库</button></div>
      {assets.length ? <div className="library-grid">{assets.map((asset) => <article className="library-card" key={asset.id}><div><img src={assetUrl(asset)} alt="" /></div><strong>{asset.original_name}</strong><span>{formatBytes(asset.size_bytes)}</span></article>)}</div> : <div className="compact-empty"><ImageIcon size={30} /><div><strong>还没有应用素材</strong><span>这里存放你主动导入、可用于穿搭与分析的图片副本。</span></div><button className="secondary-button" onClick={onImport}>导入图片</button></div>}
    </>}

    {tab === "classified" && <section>
      <div className="uncategorized-heading"><div><span>CLASSIFIED</span><h2>分类图库</h2><p>展示人工确认或本地模型建议的安全分类，可随时调整或取消。</p></div><button className="secondary-button" onClick={() => reloadAlbum()}><RefreshCw size={16} />刷新</button></div>
      {classified.length ? <div className="classified-groups">{categories.map((category) => {
        const photos = classified.filter((photo) => photo.category_id === category.id);
        if (!photos.length) return null;
        return <section key={category.id}><div className="classified-group-heading"><h3>{category.label}</h3><span>{photos.length} 张</span></div><div className="device-photo-grid">{photos.map((photo) => <article key={photo.id}>
          <div className="device-photo-preview">{photo.thumbnail_path ? <img src={devicePhotoThumbnailUrl(photo)} alt="" /> : <ImageIcon size={28} />}</div>
          <div className="device-photo-copy"><strong title={photo.file_name}>{photo.file_name}</strong><span>{formatBytes(photo.size_bytes)} · {photo.classification_source === "local_model" ? `本地模型 ${Math.round((photo.confidence ?? 0) * 100)}%` : "人工分类"}</span></div>
          <select aria-label={`调整 ${photo.file_name} 的分类`} value={photo.category_id ?? ""} disabled={assigningPhoto === photo.id} onChange={(event) => assignCategory(photo, event.target.value || null)}>
            <option value="">取消分类</option>{categories.filter((item) => item.enabled).map((item) => <option key={item.id} value={item.id}>{item.label}</option>)}
          </select>
          {tryOnActions(photo)}
        </article>)}</div></section>;
      })}</div> : <div className="large-empty"><ImageIcon size={42} /><h2>还没有已分类图片</h2><p>可在未分类图库中手动归类，或运行本地自动分类。</p></div>}
    </section>}

    {tab === "categories" && <div className="category-layout">
      <section className="category-panel">
        <div className="panel-heading"><div><span>本地自动分类</span><h3>选择你需要的分类</h3></div><small>{categories.filter((item) => item.enabled).length} 个已启用</small></div>
        <p className="panel-copy">模型只会在选中的安全类别中给出建议。低置信度、未启用类别和无法判断的图片都会进入“未分类图库”。</p>
        <div className="classifier-status"><span className={classifier?.installed ? "pill pill--ok" : "pill"}>{classifier?.installed ? "本地模型可用" : "模型未安装"}</span><small>{classifier?.engine ?? "正在检查…"}{classifier?.size_bytes ? ` · ${formatBytes(classifier.size_bytes)}` : ""} · 自动归类阈值 {Math.round((classifier?.threshold ?? 0.35) * 100)}%</small></div>
        <div className="category-grid">{categories.map((category) => <button key={category.id} role="checkbox" aria-checked={category.enabled} className={category.enabled ? "category-option is-active" : "category-option"} onClick={() => setCategories((current) => current.map((item) => item.id === category.id ? { ...item, enabled: !item.enabled } : item))}>
          <span>{category.enabled && <Check size={15} />}</span><div><strong>{category.label}</strong><small>{category.description}</small></div>
        </button>)}</div>
        <button className="primary-button category-save" onClick={saveCategories} disabled={savingCategories}>{savingCategories ? <LoaderCircle className="spin" size={17} /> : <Check size={17} />}保存分类选择</button>
      </section>
      <aside className="privacy-card"><ShieldCheck size={25} /><span>敏感信息保护</span><h3>这些内容不做自动推断</h3><p>人脸身份、民族、健康状况、宗教或政治倾向、证件与财务内容、聊天文本、精确位置。</p><small>首版不做人脸聚类，不读取 OCR 文本，不使用 EXIF 定位进行分类，也不会把整套相册交给第三方 API。</small></aside>
    </div>}

    {tab === "uncategorized" && <section>
      <div className="uncategorized-heading"><div><span>UNCATEGORIZED</span><h2>未分类图库</h2><p>新索引、低置信度或不属于所选类别的图片都会保留在这里。</p></div><div className="source-actions"><button className="primary-button" onClick={runAutoClassification} disabled={!classifier?.installed || classificationTaskId !== null || !uncategorized.length}>{classificationTaskId ? <LoaderCircle className="spin" size={16} /> : <Sparkles size={16} />}{classificationTaskId ? "正在本机分类" : "运行本地自动分类"}</button><button className="secondary-button" onClick={() => reloadAlbum()}><RefreshCw size={16} />刷新</button></div></div>
      {!!uncategorized.length && <div className="bulk-classify-bar"><button className="text-button" onClick={() => setSelectedPhotos(selectedPhotos.size === uncategorized.length ? new Set() : new Set(uncategorized.map((photo) => photo.id)))}>{selectedPhotos.size === uncategorized.length ? "取消全选" : "选择当前全部"}</button><span>已选择 {selectedPhotos.size} 张</span><select aria-label="批量分类" value={bulkCategory} onChange={(event) => setBulkCategory(event.target.value)}><option value="">选择目标分类…</option>{categories.filter((item) => item.enabled).map((category) => <option key={category.id} value={category.id}>{category.label}</option>)}</select><button className="secondary-button" onClick={applyBulkCategory} disabled={!selectedPhotos.size || !bulkCategory || bulkSaving}>{bulkSaving ? <LoaderCircle className="spin" size={15} /> : <Check size={15} />}批量归类</button></div>}
      {uncategorized.length ? <div className="device-photo-grid">{uncategorized.map((photo) => <article key={photo.id}>
        <label className="photo-select"><input type="checkbox" checked={selectedPhotos.has(photo.id)} onChange={() => toggleSelectedPhoto(photo.id)} />选择</label>
        <div className="device-photo-preview">{photo.thumbnail_path ? <img src={devicePhotoThumbnailUrl(photo)} alt="" /> : <ImageIcon size={28} />}</div>
        <div className="device-photo-copy"><strong title={photo.file_name}>{photo.file_name}</strong><span>{formatBytes(photo.size_bytes)}</span></div>
        <select aria-label={`设置 ${photo.file_name} 的分类`} value="" disabled={assigningPhoto === photo.id} onChange={(event) => assignCategory(photo, event.target.value || null)}>
          <option value="">{assigningPhoto === photo.id ? "正在保存…" : "归入分类…"}</option>
          {categories.filter((item) => item.enabled).map((category) => <option key={category.id} value={category.id}>{category.label}</option>)}
        </select>
        {tryOnActions(photo)}
      </article>)}</div> : <div className="large-empty"><Check size={42} /><h2>当前没有未分类图片</h2><p>添加设备照片目录后，新图片会先出现在这里。</p></div>}
    </section>}
  </div>;
}

function LaboratoryWorkspace({ onLibraryChanged }: { onLibraryChanged: () => Promise<void> }) {
  const [asset, setAsset] = useState<AssetSummary | null>(null);
  const [history, setHistory] = useState<LabAnalysis[]>([]);
  const [analysis, setAnalysis] = useState<LabAnalysis | null>(null);
  const [thresholdScale, setThresholdScale] = useState(1);
  const [minArea, setMinArea] = useState(40);
  const [excludeEdges, setExcludeEdges] = useState(true);
  const [scaleText, setScaleText] = useState("");
  const [showOverlay, setShowOverlay] = useState(true);
  const [importing, setImporting] = useState(false);
  const [running, setRunning] = useState(false);
  const [reviewing, setReviewing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    listLabAnalyses().then((items) => {
      setHistory(items);
      if (items[0]) {
        setAnalysis(items[0]);
        setAsset(items[0].asset);
        setThresholdScale(items[0].result.parameters.threshold_scale);
        setMinArea(items[0].result.parameters.min_area);
        setExcludeEdges(items[0].result.parameters.exclude_edges);
        setScaleText(items[0].result.parameters.scale_um_per_px?.toString() ?? "");
      }
    }).catch((reason) => setError(reason instanceof Error ? reason.message : String(reason)));
  }, []);

  const chooseLabImage = async () => {
    setImporting(true);
    setError(null);
    try {
      const imported = await chooseAndImportAsset();
      if (!imported) return;
      setAsset(imported);
      setAnalysis(null);
      await onLibraryChanged();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setImporting(false);
    }
  };

  const run = async () => {
    if (!asset) return;
    const parsedScale = scaleText.trim() ? Number(scaleText) : null;
    if (parsedScale !== null && (!Number.isFinite(parsedScale) || parsedScale <= 0)) {
      setError("像素尺度需要填写大于 0 的数字，或留空使用像素单位");
      return;
    }
    setRunning(true);
    setError(null);
    try {
      const result = await runLabSegmentation(asset.id, thresholdScale, minArea, excludeEdges, parsedScale);
      setAnalysis(result);
      setHistory((current) => [result, ...current.filter((item) => item.id !== result.id)]);
      setShowOverlay(true);
      await onLibraryChanged();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setRunning(false);
    }
  };

  const markReviewed = async () => {
    if (!analysis) return;
    setReviewing(true);
    setError(null);
    try {
      const updated = await reviewLabAnalysis(analysis.id);
      setAnalysis(updated);
      setHistory((current) => current.map((item) => item.id === updated.id ? updated : item));
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setReviewing(false);
    }
  };

  const openAnalysis = (item: LabAnalysis) => {
    setAnalysis(item);
    setAsset(item.asset);
    setThresholdScale(item.result.parameters.threshold_scale);
    setMinArea(item.result.parameters.min_area);
    setExcludeEdges(item.result.parameters.exclude_edges);
    setScaleText(item.result.parameters.scale_um_per_px?.toString() ?? "");
    setShowOverlay(true);
  };

  const summary = analysis?.result.summary;
  return <div className="workspace-view lab-workspace">
    <div className="workspace-heading"><div><span>MICRO ANALYSIS</span><h1>二维核图分析</h1><p>用可复现的 CPU 阈值与分水岭基线分割清晰核图，并保留参数和尺度口径。</p></div><span className="tryon-stage tryon-stage--ready"><span />无需下载模型</span></div>
    {error && <div className="inline-error"><CircleAlert size={17} />{error}</div>}
    <div className="lab-layout">
      <aside className="lab-history-panel">
        <div className="plant-section-title"><div><span>ANALYSES</span><h2>分析历史</h2></div><b>{history.length}</b></div>
        <button className="plant-new-button" onClick={() => { setAsset(null); setAnalysis(null); }}><ImagePlus size={16} />新建分析</button>
        <div className="lab-history-list">{history.map((item) => <button key={item.id} className={analysis?.id === item.id ? "is-selected" : ""} onClick={() => openAnalysis(item)}>
          <img src={labOverlayUrl(item)} alt="" /><span><strong>{item.asset.original_name}</strong><small>{item.result.summary.count} 个核 · {new Date(item.created_at).toLocaleDateString("zh-CN")}</small></span><i>{item.review_status === "reviewed" ? "已复核" : "未复核"}</i>
        </button>)}</div>
      </aside>

      <section className="lab-canvas-panel">
        <div className="plant-section-title"><div><span>IMAGE</span><h2>原图与轮廓</h2></div>{analysis && <div className="lab-view-toggle"><button className={!showOverlay ? "is-selected" : ""} onClick={() => setShowOverlay(false)}>原图</button><button className={showOverlay ? "is-selected" : ""} onClick={() => setShowOverlay(true)}>轮廓</button></div>}</div>
        <button className={asset ? "lab-image-canvas has-image" : "lab-image-canvas"} onClick={!asset ? chooseLabImage : undefined} disabled={importing || running}>
          {asset ? <img src={analysis && showOverlay ? labOverlayUrl(analysis) : assetUrl(asset)} alt={showOverlay ? "核分割轮廓" : "实验原图"} /> : <div><Beaker size={34} /><strong>{importing ? "正在导入…" : "选择二维核图"}</strong><span>PNG / JPEG / 单页 TIFF，最大 4096 × 4096</span></div>}
        </button>
        {asset && <div className="lab-file-row"><div><strong>{asset.original_name}</strong><span>{analysis ? `${analysis.result.metadata.width} × ${analysis.result.metadata.height} · ${analysis.result.metadata.source_dtype}` : formatBytes(asset.size_bytes)}</span></div><button onClick={chooseLabImage}>更换图像</button></div>}
        {analysis && <div className="lab-object-table"><div><strong>对象测量预览</strong><span>前 {Math.min(analysis.result.measurements.length, 8)} / {analysis.result.measurements.length}</span></div><table><thead><tr><th>ID</th><th>面积</th><th>等效直径</th><th>质心 x, y</th></tr></thead><tbody>{analysis.result.measurements.slice(0, 8).map((item) => <tr key={item.object_id}><td>{item.object_id}</td><td>{item.area.toFixed(1)} {summary?.area_unit}</td><td>{item.equivalent_diameter.toFixed(1)} {summary?.unit}</td><td>{item.centroid_x}, {item.centroid_y}</td></tr>)}</tbody></table></div>}
      </section>

      <aside className="lab-controls-panel">
        <div className="plant-section-title"><div><span>PARAMETERS</span><h2>分割与测量</h2></div><small>对象：细胞核</small></div>
        <div className="lab-scope-warning"><CircleAlert size={16} /><span>核数量不等于细胞数量；当前自动结果需人工查看轮廓。</span></div>
        <label className="lab-control"><span><strong>阈值系数</strong><b>{thresholdScale.toFixed(2)}</b></span><input type="range" min="0.5" max="1.5" step="0.05" value={thresholdScale} onChange={(event) => setThresholdScale(Number(event.target.value))} /><small>调高会保留更亮区域，通常减少检测对象。</small></label>
        <label className="lab-control"><span><strong>最小核面积</strong><b>{minArea} px²</b></span><input type="range" min="4" max="500" step="4" value={minArea} onChange={(event) => setMinArea(Number(event.target.value))} /><small>小于该面积的亮点按碎片排除。</small></label>
        <label className="lab-scale"><span>像素尺度 <small>可选</small></span><div><input inputMode="decimal" value={scaleText} onChange={(event) => setScaleText(event.target.value)} placeholder="例如 0.325" /><b>μm/px</b></div><small>仅填写可靠标定值；放大倍数本身不能换算尺度。</small></label>
        <label className="lab-checkbox"><input type="checkbox" checked={excludeEdges} onChange={(event) => setExcludeEdges(event.target.checked)} /><span><strong>排除触碰边缘的对象</strong><small>避免把被图像边界截断的核纳入统计。</small></span></label>
        <button className="generate-button" onClick={run} disabled={!asset || running}>{running ? <LoaderCircle className="spin" size={18} /> : <Beaker size={18} />}{!asset ? "请先选择图像" : running ? "正在 CPU 分割与测量" : analysis ? "用当前参数重新分析" : "运行二维核分割"}</button>
        {analysis && <div className="lab-summary">
          <div><span>有效核数量</span><strong>{summary?.count ?? 0}</strong><small>二维实例</small></div>
          <div><span>平均面积</span><strong>{summary?.mean_area?.toFixed(1) ?? "N/A"}</strong><small>{summary?.area_unit}</small></div>
          <div><span>平均等效直径</span><strong>{summary?.mean_equivalent_diameter?.toFixed(1) ?? "N/A"}</strong><small>{summary?.unit}</small></div>
        </div>}
        {analysis && <button className={analysis.review_status === "reviewed" ? "lab-reviewed" : "secondary-button lab-review"} onClick={markReviewed} disabled={reviewing || analysis.review_status === "reviewed"}>{reviewing ? <LoaderCircle className="spin" size={15} /> : <Check size={15} />}{analysis.review_status === "reviewed" ? "已人工复核当前轮廓" : "确认已人工复核轮廓"}</button>}
        {analysis && <div className="lab-provenance"><span>{analysis.result.algorithm_revision}</span><span>CPU · {analysis.result.elapsed_seconds.toFixed(2)} 秒</span><span>{analysis.result.parameters.scale_um_per_px ? "物理尺度" : "仅像素单位"}</span></div>}
      </aside>
    </div>
  </div>;
}

function PlantWorkspace({ modelStatus, onLibraryChanged, onNavigate }: {
  modelStatus: PlantModelStatus | null;
  onLibraryChanged: () => Promise<void>;
  onNavigate: (id: WorkspaceId) => void;
}) {
  const [asset, setAsset] = useState<AssetSummary | null>(null);
  const [organ, setOrgan] = useState<PlantObservation["organ"]>("whole");
  const [note, setNote] = useState("");
  const [history, setHistory] = useState<PlantObservation[]>([]);
  const [observation, setObservation] = useState<PlantObservation | null>(null);
  const [importing, setImporting] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);
  const [confirming, setConfirming] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    listPlantObservations()
      .then((items) => {
        setHistory(items);
        if (items[0]) {
          setObservation(items[0]);
          setAsset(items[0].asset);
          setOrgan(items[0].organ);
          setNote(items[0].note);
        }
      })
      .catch((reason) => setError(reason instanceof Error ? reason.message : String(reason)));
  }, []);

  const choosePlantImage = async () => {
    setImporting(true);
    setError(null);
    try {
      const imported = await chooseAndImportAsset();
      if (!imported) return;
      setAsset(imported);
      setObservation(null);
      setNote("");
      await onLibraryChanged();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setImporting(false);
    }
  };

  const analyze = async () => {
    if (!asset || !modelStatus?.ready) return;
    setAnalyzing(true);
    setError(null);
    try {
      const result = await identifyPlant(asset.id, organ, note.trim());
      setObservation(result);
      setHistory((current) => [result, ...current.filter((item) => item.id !== result.id)]);
      await onLibraryChanged();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setAnalyzing(false);
    }
  };

  const confirm = async (taxonId: string | null) => {
    if (!observation) return;
    setConfirming(true);
    setError(null);
    try {
      const updated = await confirmPlantObservation(observation.id, taxonId);
      setObservation(updated);
      setHistory((current) => current.map((item) => item.id === updated.id ? updated : item));
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setConfirming(false);
    }
  };

  const openObservation = (item: PlantObservation) => {
    setObservation(item);
    setAsset(item.asset);
    setOrgan(item.organ);
    setNote(item.note);
    setError(null);
  };

  const organLabels: Record<PlantObservation["organ"], string> = {
    whole: "整株", leaf: "叶片", flower: "花", fruit: "果实",
  };
  const ready = Boolean(modelStatus?.ready);

  return <div className="workspace-view plant-workspace">
    <div className="workspace-heading">
      <div><span>PLANT LAB</span><h1>植物观察</h1><p>在固定 20 种范围内生成本地候选，由你确认后写入观察档案。</p></div>
      <span className={`tryon-stage ${ready ? "tryon-stage--ready" : ""}`}><span />{ready ? "BioCLIP 本地可用" : "等待植物模型"}</span>
    </div>
    {error && <div className="inline-error"><CircleAlert size={17} />{error}</div>}
    <div className="plant-layout">
      <aside className="plant-history-panel">
        <div className="plant-section-title"><div><span>OBSERVATIONS</span><h2>观察档案</h2></div><b>{history.length}</b></div>
        <button className="plant-new-button" onClick={() => { setAsset(null); setObservation(null); setNote(""); setOrgan("whole"); }}><ImagePlus size={16} />新建观察</button>
        <div className="plant-history-list">
          {history.map((item) => <button key={item.id} className={observation?.id === item.id ? "is-selected" : ""} onClick={() => openObservation(item)}>
            <img src={assetUrl(item.asset)} alt="" />
            <span><strong>{item.confirmed_chinese_name ?? "待确认植物"}</strong><small>{organLabels[item.organ]} · {new Date(item.created_at).toLocaleDateString("zh-CN")}</small></span>
            <i className={`plant-status plant-status--${item.status}`}>{item.status === "confirmed" ? "已确认" : item.status === "unresolved" ? "未确定" : "待确认"}</i>
          </button>)}
          {!history.length && <div className="plant-history-empty"><Leaf size={24} /><span>分析后的观察会保存在这里</span></div>}
        </div>
      </aside>

      <section className="plant-input-panel">
        <div className="plant-section-title"><div><span>STEP 01</span><h2>观察图片</h2></div><small>不读取精确位置</small></div>
        <button className={asset ? "plant-image-input has-image" : "plant-image-input"} onClick={choosePlantImage} disabled={importing || analyzing}>
          {asset ? <img src={assetUrl(asset)} alt="植物观察输入" /> : <div><ImagePlus size={30} /><strong>{importing ? "正在导入…" : "选择植物照片"}</strong><span>建议主体清晰，并补拍叶、花或果实</span></div>}
        </button>
        {asset && <div className="plant-file-row"><span><strong>{asset.original_name}</strong><small>{formatBytes(asset.size_bytes)} · 已复制到本地素材库</small></span><button onClick={choosePlantImage}>更换图片</button></div>}
        <div className="plant-field"><label>图片主要部位</label><div className="choice-row">{(Object.keys(organLabels) as PlantObservation["organ"][]).map((item) => <button key={item} className={organ === item ? "is-selected" : ""} onClick={() => { setOrgan(item); setObservation(null); }}>{organLabels[item]}</button>)}</div></div>
        <label className="plant-note"><span>观察备注 <small>可选，不用于自动推断身份或位置</small></span><textarea maxLength={500} value={note} onChange={(event) => setNote(event.target.value)} placeholder="例如：阳台盆栽，今天首次开花…" /></label>
        <button className="generate-button plant-analyze" onClick={analyze} disabled={!asset || !ready || analyzing}>
          {analyzing ? <LoaderCircle className="spin" size={18} /> : <Leaf size={18} />}
          {!asset ? "请先选择植物照片" : !ready ? "植物模型尚未就绪" : analyzing ? "正在本机检索候选" : "生成本地识别候选"}
        </button>
        {!ready && <button className="model-link" onClick={() => onNavigate("models")}>前往模型中心查看状态<ChevronRight size={15} /></button>}
        <div className="local-note"><CloudOff size={16} /><span>图片仅在本机分析；当前不会上传服务器，也不会读取 EXIF 定位。</span></div>
      </section>

      <section className="plant-result-panel">
        <div className="plant-section-title"><div><span>STEP 02</span><h2>候选与确认</h2></div>{observation && <i className={`plant-status plant-status--${observation.status}`}>{observation.status === "confirmed" ? "已确认" : observation.status === "unresolved" ? "暂不确认" : "等待确认"}</i>}</div>
        {observation ? <>
          <div className="plant-caution"><CircleAlert size={18} /><div><strong>{observation.prediction.notice}</strong><span>分数只表示这 20 种之间的相似排序，未经校准，不能当作准确率。</span></div></div>
          <div className="plant-candidates">{observation.prediction.candidates.map((candidate, index) => <article key={candidate.taxon_id} className={observation.confirmed_taxon_id === candidate.taxon_id ? "is-confirmed" : ""}>
            <b>0{index + 1}</b><div><strong>{candidate.chinese_name}</strong><em>{candidate.scientific_name}</em><span>范围内相似分 {Math.round(candidate.score * 1000) / 10}</span></div>
            <button onClick={() => confirm(candidate.taxon_id)} disabled={confirming}>{observation.confirmed_taxon_id === candidate.taxon_id ? <><Check size={14} />已确认</> : "确认此候选"}</button>
          </article>)}</div>
          <div className="plant-evidence"><strong>仍缺少的证据</strong>{observation.prediction.missing_evidence.map((item) => <span key={item}><CircleAlert size={12} />{item}</span>)}</div>
          <button className="plant-unresolved" onClick={() => confirm(null)} disabled={confirming}>{observation.status === "unresolved" ? <Check size={15} /> : <CircleAlert size={15} />}{observation.status === "unresolved" ? "已记录为暂不确认" : "都不像，暂不确认"}</button>
          <div className="plant-runtime"><span>{observation.prediction.width} × {observation.prediction.height}</span><span>{observation.prediction.device.toUpperCase()}</span><span>{observation.prediction.elapsed_seconds.toFixed(1)} 秒</span></div>
        </> : <div className="plant-result-empty"><Leaf size={38} /><h3>等待本地候选</h3><p>选择一张照片并注明主要部位。模型只检索首版支持的 20 种常见植物。</p><small>不会判断可食用性、毒性、药用价值或病因。</small></div>}
      </section>
    </div>
  </div>;
}

function TryOnWorkspace({ modelStatus, onLibraryChanged, onNavigate }: {
  modelStatus: TryOnModelStatus | null;
  onLibraryChanged: () => Promise<void>;
  onNavigate: (id: WorkspaceId) => void;
}) {
  const [person, setPerson] = useState<AssetSummary | null>(null);
  const [garment, setGarment] = useState<AssetSummary | null>(null);
  const [importingSlot, setImportingSlot] = useState<"person" | "garment" | null>(null);
  const [draftLoading, setDraftLoading] = useState(true);
  const [preflight, setPreflight] = useState<TryOnPreflightResult | null>(null);
  const [preflightRunning, setPreflightRunning] = useState(false);
  const [category, setCategory] = useState<"automatic" | "tops" | "bottoms" | "one-pieces">("automatic");
  const [garmentPhotoType, setGarmentPhotoType] = useState<"model" | "flat-lay">("model");
  const [sleeveMode, setSleeveMode] = useState<"automatic" | "standard" | "reveal-arms">("automatic");
  const [numTimesteps, setNumTimesteps] = useState<20 | 30>(20);
  const [generation, setGeneration] = useState<TryOnHistoryItem | null>(null);
  const [history, setHistory] = useState<TryOnHistoryItem[]>([]);
  const [comparePosition, setComparePosition] = useState(55);
  const [generating, setGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([loadTryOnDraft(), listTryOnResults()])
      .then(([draft, storedHistory]) => {
        setPerson(draft.person);
        setGarment(draft.garment);
        setCategory(draft.category);
        setGarmentPhotoType(draft.garment_photo_type);
        setSleeveMode(draft.sleeve_mode);
        setHistory(storedHistory);
        setGeneration(storedHistory[0] ?? null);
      })
      .catch((reason) => setError(reason instanceof Error ? reason.message : String(reason)))
      .finally(() => setDraftLoading(false));
  }, []);

  const chooseInput = async (slot: "person" | "garment") => {
    setImportingSlot(slot);
    setError(null);
    try {
      const asset = await chooseAndImportAsset();
      if (!asset) return;
      const nextPerson = slot === "person" ? asset : person;
      const nextGarment = slot === "garment" ? asset : garment;
      setPerson(nextPerson);
      setGarment(nextGarment);
      setPreflight(null);
      setGeneration(null);
      await saveTryOnDraft(nextPerson?.id ?? null, nextGarment?.id ?? null, category, garmentPhotoType, sleeveMode);
      await onLibraryChanged();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setImportingSlot(null);
    }
  };

  const hasInputs = Boolean(person && garment);
  const modelReady = Boolean(modelStatus?.ready);

  const updateOptions = async (
    nextCategory: "automatic" | "tops" | "bottoms" | "one-pieces",
    nextGarmentPhotoType: "model" | "flat-lay",
    nextSleeveMode: "automatic" | "standard" | "reveal-arms" = sleeveMode,
  ) => {
    const compatibleSleeveMode = nextSleeveMode;
    setCategory(nextCategory);
    setGarmentPhotoType(nextGarmentPhotoType);
    setSleeveMode(compatibleSleeveMode);
    setPreflight(null);
    if (compatibleSleeveMode === "reveal-arms") setNumTimesteps(30);
    setGeneration(null);
    try {
      await saveTryOnDraft(person?.id ?? null, garment?.id ?? null, nextCategory, nextGarmentPhotoType, compatibleSleeveMode);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    }
  };

  const runPreflight = async () => {
    if (!person || !garment) return;
    setPreflightRunning(true);
    setError(null);
    try {
      setPreflight(await runTryOnPreflight(person.id, garment.id, garmentPhotoType));
      await onLibraryChanged();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setPreflightRunning(false);
    }
  };

  const generateTryOn = async () => {
    if (!person || !garment || !preflight || preflight.status === "block" || !modelReady) return;
    setGenerating(true);
    setError(null);
    setGeneration(null);
    try {
      const result = await runTryOnGeneration(person.id, garment.id, category, garmentPhotoType, sleeveMode, numTimesteps);
      await onLibraryChanged();
      const storedHistory = await listTryOnResults();
      setHistory(storedHistory);
      setGeneration(storedHistory.find((item) => item.task_id === result.task_id) ?? storedHistory[0] ?? null);
      setComparePosition(55);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setGenerating(false);
    }
  };

  return <div className="workspace-view tryon-workspace">
    <div className="workspace-heading">
      <div><span>TRY-ON STUDIO</span><h1>虚拟穿搭</h1><p>人物图和服装图只在本机处理，结果自动加入本地素材库。</p></div>
      <span className={`tryon-stage ${modelReady ? "tryon-stage--ready" : ""}`}><span />{modelReady ? "本地模型已就绪" : "等待本地模型"}</span>
    </div>

    {error && <div className="inline-error"><CircleAlert size={17} />{error}</div>}

    <div className="tryon-layout">
      <section className="tryon-panel tryon-inputs">
        <div className="tryon-panel-heading"><div><span>STEP 01</span><h2>准备输入</h2></div><small>图片仅复制到本地素材库</small></div>
        <div className="tryon-slot-grid">
          <TryOnSlot
            label="人物照片"
            hint="单人、正面或微侧、上半身清晰"
            asset={person}
            loading={draftLoading || importingSlot === "person"}
            onChoose={() => chooseInput("person")}
          />
          <TryOnSlot
            label="服装照片"
            hint="单件上衣、平铺或人台、背景简洁"
            asset={garment}
            loading={draftLoading || importingSlot === "garment"}
            onChoose={() => chooseInput("garment")}
          />
        </div>

        <div className="tryon-option">
          <div><strong>服装类别</strong><span>默认由本地人体解析判断；不确定时可手动指定</span></div>
          <div className="choice-row">
            <button className={category === "automatic" ? "is-selected" : ""} onClick={() => updateOptions("automatic", garmentPhotoType)}>智能判断</button>
            <button className={category === "tops" ? "is-selected" : ""} onClick={() => updateOptions("tops", garmentPhotoType)}>上衣</button>
            <button className={category === "one-pieces" ? "is-selected" : ""} onClick={() => updateOptions("one-pieces", garmentPhotoType)}>连衣裙</button>
            <button className={category === "bottoms" ? "is-selected" : ""} onClick={() => updateOptions("bottoms", garmentPhotoType)}>下装</button>
          </div>
        </div>

        <div className="tryon-option compact-option">
          <div><strong>服装图片</strong><span>选择最接近原图的形式</span></div>
          <div className="choice-row">
            <button className={garmentPhotoType === "model" ? "is-selected" : ""} onClick={() => updateOptions(category, "model")}>模特穿着</button>
            <button className={garmentPhotoType === "flat-lay" ? "is-selected" : ""} onClick={() => updateOptions(category, "flat-lay")}>平铺商品图</button>
          </div>
        </div>

        <div className="tryon-option sleeve-option">
          <div>
            <strong>露出区域与遮挡</strong>
            <span>默认分析袖长、裤长、可见身体区域与遮挡，再决定保持或补全</span>
          </div>
          <div className="choice-row">
            <button className={sleeveMode === "automatic" ? "is-selected" : ""} onClick={() => updateOptions(category, garmentPhotoType, "automatic")}>智能判断</button>
            <button className={sleeveMode === "standard" ? "is-selected" : ""} onClick={() => updateOptions(category, garmentPhotoType, "standard")}>常规换装</button>
            <button
              className={sleeveMode === "reveal-arms" ? "is-selected" : ""}
              onClick={() => updateOptions(category, garmentPhotoType, "reveal-arms")}
            >强制补全 · 实验</button>
          </div>
        </div>
        {sleeveMode === "automatic" && <div className="adaptive-note">
          <ShieldCheck size={18} />
          <div><strong>本地模型会先做人体与服装解析</strong><span>目标服装露出手臂或腿部、原图对应区域被遮住且姿态清楚时，会启用补全；手机和手持物只会局部保留，不会阻止另一侧手臂或腿部生成。</span></div>
        </div>}
        {sleeveMode === "reveal-arms" && <div className="occlusion-warning">
          <CircleAlert size={18} />
          <div><strong>需要露出的手臂或腿部将由本地模型推测生成</strong><span>这不是对真实皮肤、纹身或饰品的还原。正面站立、四肢与躯干分开且双手可见时效果更稳定。</span></div>
        </div>}

        <div className="tryon-option compact-option">
          <div><strong>生成质量</strong><span>20 步约半分钟，30 步细节更稳</span></div>
          <div className="choice-row">
            <button className={numTimesteps === 20 ? "is-selected" : ""} onClick={() => setNumTimesteps(20)}>快速</button>
            <button className={numTimesteps === 30 ? "is-selected" : ""} onClick={() => setNumTimesteps(30)}>标准</button>
          </div>
        </div>

        <div className="quality-guide">
          <ShieldCheck size={19} />
          <div><strong>先运行本地图片预检</strong><span>检查格式、尺寸、曝光、清晰度和单人候选；警告项允许继续，阻止项需要先修正。</span></div>
        </div>

        <button className="generate-button" onClick={runPreflight} disabled={!hasInputs || preflightRunning || draftLoading}>
          {preflightRunning ? <LoaderCircle className="spin" size={18} /> : <ShieldCheck size={18} />}
          {!hasInputs ? "请先准备两张图片" : preflightRunning ? "正在本地检查" : preflight ? "重新检查两张图片" : "检查两张图片"}
        </button>
        {preflight && <button className={preflight.status !== "block" && modelReady ? "generate-button generation-button" : "generation-gate"} onClick={generateTryOn} disabled={preflight.status === "block" || !modelReady || generating}>
          {generating ? <LoaderCircle className="spin" size={17} /> : <WandSparkles size={17} />}
          {preflight.status === "block" ? "存在阻止项，请更换输入" : !modelReady ? "预检完成，试衣模型尚未安装" : generating ? "正在本机生成，请稍候" : generation ? "重新生成穿搭结果" : "开始本地生成"}
        </button>}
        {preflight && !modelReady && <button className="model-link" onClick={() => onNavigate("models")}>前往模型中心查看状态<ChevronRight size={15} /></button>}
      </section>

      <section className="tryon-panel result-panel">
        <div className="tryon-panel-heading"><div><span>STEP 02</span><h2>{generation ? "生成结果" : preflight ? "输入检查" : "生成结果"}</h2></div><span className={`pill ${generation ? `pill--${generation.verification_status === "passed" ? "pass" : "warning"}` : preflight ? `pill--${preflight.status}` : ""}`}>{generation ? (generation.verification_status === "passed" ? "核验通过" : "未通过核验") : preflight ? preflightStatusLabel(preflight.status) : "等待输入"}</span></div>
        {generation ? <div className="tryon-generated-result">
          <div className="tryon-compare-stage">
            <img src={assetUrl(generation.person_asset)} alt="人物原图" />
            <img className="tryon-compare-output" style={{ clipPath: `inset(0 0 0 ${comparePosition}%)` }} src={tryOnResultUrl(generation)} alt="虚拟穿搭生成结果" />
            <span className="compare-divider" style={{ left: `${comparePosition}%` }} />
            <span className="compare-label compare-label--before">原图</span>
            <span className="compare-label compare-label--after">结果</span>
          </div>
          <label className="compare-control"><span>拖动对比</span><input aria-label="原图与生成结果对比" type="range" min="0" max="100" value={comparePosition} onChange={(event) => setComparePosition(Number(event.target.value))} /></label>
          <div><strong>{generation.verification_status === "passed" ? "已通过本地质量核验" : "已生成，建议人工检查"}</strong><span>{generation.width} × {generation.height} · {generation.num_timesteps} 步{generation.generation_passes === 2 ? " × 试穿局部修补" : " × 单阶段"}{generation.hand_refinement_used ? " × 独立手部修复" : ""} · {generation.elapsed_seconds.toFixed(1)} 秒 · {generation.quality_checks?.garment?.message ?? "已保存到本地素材库"}</span></div>
          {generation.requested_category === "automatic" && <small className="result-coverage-note">本地服装解析已判断为：{generation.category === "one-pieces" ? "连衣裙" : generation.category === "tops" ? "上衣" : "下装"}。{generation.category_notice}</small>}
          <small className="result-coverage-note">{generation.coverage_notice}</small>
          {generation.quality_checks && <div className="result-coverage-note">
            <span>逐侧核验：{generation.quality_checks.limbs.arms.map((arm) => `${arm.side === "left" ? "画面左侧" : "画面右侧"}上臂 ${Math.round(arm.upper_arm_coverage * 100)}%、前臂 ${Math.round(arm.forearm_coverage * 100)}%、手${arm.hand_detected ? `完整（${arm.hand_supported_points || arm.hand_keypoints || "已识别"}点，清晰度 ${Math.round(arm.hand_detail_score || 0)}）` : "未通过"}${arm.source_hand_occluded ? "，原手遮挡" : "，原手可见"}`).join("；")}。</span>
            <span>肤色参考：{Object.values(generation.quality_checks.skin_tone.reference_modes ?? {}).some((mode) => mode === "same-position") ? "优先采用原图同侧同位置光照" : "原位置无可见皮肤，已使用同侧或面部参考"}；差异 {generation.skin_tone_delta_before.toFixed(1)} → {generation.skin_tone_delta_after.toFixed(1)}。</span>
          </div>}
        </div> : preflight ? <PreflightResults result={preflight} /> : <div className="result-canvas">
          <div className="result-mark"><WandSparkles size={28} /></div>
          <strong>这里将显示本地生成结果</strong>
          <p>本机已验证 576 × 864 输出；准备两张图片并完成预检后即可生成。</p>
        </div>}
        <div className="pipeline-row">
          <div><b>01</b><span>输入预检</span></div><ChevronRight size={14} />
          <div><b>02</b><span>人体与服装条件</span></div><ChevronRight size={14} />
          <div><b>03</b><span>本地生成与核验</span></div>
        </div>
        <div className="local-note"><CloudOff size={16} /><span>当前流程不会把照片上传到服务器或第三方 API。</span></div>
        {history.length > 0 && <div className="tryon-history">
          <div><strong>历史结果</strong><span>{history.length} 个本地版本</span></div>
          <div className="tryon-history-strip">{history.map((item) => <button key={item.task_id} className={generation?.task_id === item.task_id ? "is-selected" : ""} onClick={() => { setGeneration(item); setComparePosition(55); }} title={new Date(item.created_at).toLocaleString("zh-CN")}><img src={tryOnResultUrl(item)} alt="" /><span>{new Date(item.created_at).toLocaleDateString("zh-CN", { month: "2-digit", day: "2-digit" })}</span></button>)}</div>
        </div>}
      </section>
    </div>
  </div>;
}

function TryOnSlot({ label, hint, asset, loading, onChoose }: {
  label: string;
  hint: string;
  asset: AssetSummary | null;
  loading: boolean;
  onChoose: () => void;
}) {
  return <div className={asset ? "tryon-slot has-image" : "tryon-slot"}>
    <button className="tryon-preview" onClick={onChoose} disabled={loading}>
      {asset ? <img src={assetUrl(asset)} alt={label} /> : <><span><ImagePlus size={25} /></span><strong>{loading ? "正在导入…" : `选择${label}`}</strong><small>JPEG / PNG / WebP / TIFF</small></>}
    </button>
    <div className="slot-copy"><strong>{asset?.original_name ?? label}</strong><span>{asset ? `${formatBytes(asset.size_bytes)} · 已复制到素材库` : hint}</span></div>
    {asset && <button className="replace-button" onClick={onChoose}>更换</button>}
  </div>;
}

function preflightStatusLabel(status: TryOnPreflightResult["status"]) {
  if (status === "pass") return "检查通过";
  if (status === "warning") return "建议确认";
  return "需要修正";
}

function PreflightResults({ result }: { result: TryOnPreflightResult }) {
  return <div className="preflight-results">
    <div className={`preflight-summary preflight-summary--${result.status}`}>
      {result.status === "pass" ? <Check size={20} /> : <CircleAlert size={20} />}
      <div>
        <strong>{preflightStatusLabel(result.status)}</strong>
        <span>{result.status === "block" ? "至少一项输入不能进入生成队列" : result.status === "warning" ? "可以继续评估，但建议先确认提醒项" : "两张图片通过当前本地文件级检查"}</span>
      </div>
    </div>
    <div className="preflight-columns">
      <PreflightImageCard title="人物照片" result={result.person} />
      <PreflightImageCard title="服装照片" result={result.garment} />
    </div>
    {result.scene.blockers.map((blocker) => <div className="preflight-limitation" key={blocker}><CircleAlert size={14} /><span>{blocker}</span></div>)}
    {result.scene.supported && <div className="preflight-limitation"><Check size={14} /><span>人体与服装语义检查通过；人物有效区域约占画面 {Math.round(result.scene.person_coverage * 100)}%，目标服装判断为{result.scene.category === "one-pieces" ? "连体服装" : result.scene.category === "tops" ? "上衣" : "下装"}。</span></div>}
    <div className="preflight-limitation"><CircleAlert size={14} /><span>{result.person.limitations[0]}</span></div>
  </div>;
}

function PreflightImageCard({ title, result }: { title: string; result: ImagePreflightResult }) {
  return <article className="preflight-card">
    <div className="preflight-card__heading">
      <div><strong>{title}</strong><span>{result.metrics.width} × {result.metrics.height} · {result.metrics.format}</span></div>
      <span className={`check-state check-state--${result.status}`}>{preflightStatusLabel(result.status)}</span>
    </div>
    <div className="check-list">
      {result.checks.map((item) => <div className={`check-item check-item--${item.level}`} key={item.id}>
        <span className="check-dot">{item.level === "pass" ? <Check size={11} /> : <CircleAlert size={11} />}</span>
        <div><strong>{item.label}</strong><span>{item.message}</span></div>
      </div>)}
    </div>
  </article>;
}

function TaskCenter({ tasks, onNavigate, onRefresh, onNotice }: { tasks: TaskSummary[]; onNavigate: (id: WorkspaceId) => void; onRefresh: () => Promise<void>; onNotice: (message: string | null) => void }) {
  const [cancellingTask, setCancellingTask] = useState<string | null>(null);
  useEffect(() => {
    onRefresh().catch(() => undefined);
  }, []);
  const requestCancellation = async (task: TaskSummary) => {
    setCancellingTask(task.id);
    try {
      await cancelTask(task.id);
      onNotice("已请求取消任务，应用会在安全检查点停止");
      await onRefresh();
    } catch (error) {
      onNotice(error instanceof Error ? error.message : String(error));
    } finally {
      setCancellingTask(null);
    }
  };
  return <div className="workspace-view">
    <div className="workspace-heading">
      <div><span>LOCAL TASKS</span><h1>任务中心</h1><p>任务状态和失败原因保存在本地，重启后仍可查看。</p></div>
      <button className="secondary-button" onClick={() => onNavigate("tryon")}><Shirt size={17} />返回虚拟穿搭</button>
    </div>
    {tasks.length ? <div className="task-list">
      {tasks.map((task) => {
        const successful = task.status === "succeeded";
        const running = task.status === "running" || task.status === "queued";
        return <article className="task-row" key={task.id}>
          <div className={`task-icon task-icon--${task.status}`}>
            {running ? <LoaderCircle className="spin" size={19} /> : successful ? <Check size={19} /> : <CircleAlert size={19} />}
          </div>
          <div className="task-copy">
            <div><strong>{task.task_type === "tryon.preflight" ? "虚拟穿搭输入检查" : task.task_type === "tryon.generate" ? "本地虚拟穿搭生成" : task.task_type === "album.scan" ? "设备相册扫描" : task.task_type === "album.classify" ? "本地相册自动分类" : task.task_type === "plant.identify" ? "本地植物候选检索" : task.task_type === "lab.segment" ? "二维核图分割与测量" : task.task_type}</strong><span className={`task-status task-status--${task.status}`}>{task.cancel_requested && running ? "正在取消" : taskStatusLabel(task.status)}</span></div>
            <span>{task.stage === "complete" ? (task.task_type === "album.scan" ? "索引与缩略图已完成" : task.task_type === "album.classify" ? "安全类别分类已完成" : task.task_type === "tryon.generate" ? "生成已结束，请按核验状态复查结果" : task.task_type === "plant.identify" ? "候选已保存，等待人工确认" : task.task_type === "lab.segment" ? "轮廓和二维测量已保存，等待复核" : "图片检查已完成") : task.error_code ? `错误：${task.error_code}` : task.cancel_requested ? "等待当前安全步骤结束" : task.stage === "local_generation" ? "正在加载模型并生成图片" : task.stage === "local_retrieval" ? "正在加载本地模型并检索候选" : task.stage === "segmentation" ? "正在进行 CPU 分割与测量" : task.stage === "input_checks" ? "正在复核输入图片" : `阶段：${task.stage ?? "等待处理"}`}</span>
          </div>
          <div className="task-outcome">
            {task.result_status && <span className={`check-state check-state--${task.result_status}`}>{preflightStatusLabel(task.result_status)}</span>}
            {running && <button className="task-cancel-button" onClick={() => requestCancellation(task)} disabled={task.cancel_requested || cancellingTask !== null}>{cancellingTask === task.id ? <LoaderCircle className="spin" size={13} /> : <X size={13} />}{task.cancel_requested ? "取消中" : "取消"}</button>}
            <time>{new Date(task.created_at).toLocaleString("zh-CN", { month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit" })}</time>
          </div>
        </article>;
      })}
    </div> : <div className="large-empty"><Layers3 size={42} /><h2>还没有任务</h2><p>完成一次图片预检后，记录会显示在这里。</p><button className="secondary-button" onClick={() => onNavigate("tryon")}>开始准备输入</button></div>}
  </div>;
}

function taskStatusLabel(status: TaskSummary["status"]) {
  if (status === "succeeded") return "已完成";
  if (status === "failed") return "失败";
  if (status === "cancelled") return "已取消";
  if (status === "running") return "运行中";
  return "排队中";
}

function ModelCenter({ engine, checking, onCheck, snapshot, tryOnModel, handRefiner, plantModel }: { engine: EngineHealth | null; checking: boolean; onCheck: () => void; snapshot: DashboardSnapshot; tryOnModel: TryOnModelStatus | null; handRefiner: HandRefinerStatus | null; plantModel: PlantModelStatus | null }) {
  return <div className="workspace-view">
    <div className="workspace-heading"><div><span>LOCAL MODELS</span><h1>模型中心</h1><p>模型按需安装，当前不会自动下载大型权重。</p></div><button className="secondary-button" onClick={onCheck} disabled={checking}>{checking ? <LoaderCircle className="spin" size={18} /> : <Cpu size={18} />}检查引擎</button></div>
    <div className="model-grid">
      <article className="model-card"><div className="model-card__top"><Cpu /><span className={engine?.available ? "pill pill--ok" : "pill"}>{engine?.available ? "可用" : "待检查"}</span></div><h3>基础 Python 引擎</h3><p>{engine?.message ?? "负责本地图像处理与后续模型适配。"}</p><dl><div><dt>Python</dt><dd>{engine?.python_version ?? "—"}</dd></div><div><dt>协议</dt><dd>{engine?.protocol_version ?? "—"}</dd></div></dl></article>
      <article className="model-card"><div className="model-card__top"><Shirt /><span className={tryOnModel?.ready ? "pill pill--ok" : "pill"}>{tryOnModel?.ready ? "本机可用" : "待安装"}</span></div><h3>FASHN VTON v1.5</h3><p>{tryOnModel?.ready ? "本地生成已通过完整样例测试；发布前仍需完成人体解析器许可证审查。" : tryOnModel?.message ?? "按需安装约 2.14GB 权重和独立运行时。"}</p><dl><div><dt>设备</dt><dd>{tryOnModel?.gpu_name ?? (snapshot.device.gpu_available ? "NVIDIA GPU" : "需要约 8GB GPU")}</dd></div><div><dt>已装权重</dt><dd>{tryOnModel ? formatBytes(tryOnModel.weights_bytes) : "—"}</dd></div><div><dt>姿态检测</dt><dd>{tryOnModel?.pose_provider === "CPUExecutionProvider" ? "CPU（生成仍走 GPU）" : tryOnModel?.pose_provider ?? "—"}</dd></div></dl></article>
      <article className="model-card"><div className="model-card__top"><WandSparkles /><span className={handRefiner?.ready ? "pill pill--ok" : "pill pill--muted"}>{handRefiner?.ready ? "本机可用" : "未安装"}</span></div><h3>手部局部修复</h3><p>{handRefiner?.message ?? "仅在左右手独立核验失败时运行；使用隔离环境，不会改动试穿模型。"}</p><dl><div><dt>运行方式</dt><dd>本地 · 按需 · 隔离</dd></div><div><dt>模型体积</dt><dd>{handRefiner ? formatBytes(handRefiner.expected_download_bytes) : "约 1.99 GB"}</dd></div><div><dt>当前组成</dt><dd>{handRefiner ? [handRefiner.runtime_installed, handRefiner.base_model_installed].filter(Boolean).length : 0} / 2</dd></div><div><dt>当前状态</dt><dd>{handRefiner?.ready ? "单样例已通过" : "等待安装"}</dd></div></dl></article>
      <article className="model-card"><div className="model-card__top"><Leaf /><span className={plantModel?.ready ? "pill pill--ok" : "pill"}>{plantModel?.ready ? "本机可用" : "待安装"}</span></div><h3>BioCLIP ViT-B/16</h3><p>{plantModel?.ready ? "为固定 20 种常见植物提供未校准候选，结论必须由用户确认。" : plantModel?.message ?? "按需安装约 571MB 权重和独立运行时。"}</p><dl><div><dt>支持范围</dt><dd>{plantModel?.supported_taxa ?? 20} 种</dd></div><div><dt>已装权重</dt><dd>{plantModel ? formatBytes(plantModel.weights_bytes) : "—"}</dd></div><div><dt>发布状态</dt><dd>实验功能</dd></div></dl></article>
      <article className="model-card"><div className="model-card__top"><Palette /><span className="pill pill--muted">后续</span></div><h3>视觉描述与标签</h3><p>用于补充可见特征描述和相册标签，可选择本地或免费 API。</p><dl><div><dt>默认</dt><dd>本地关闭</dd></div><div><dt>付费预算</dt><dd>¥0</dd></div></dl></article>
    </div>
  </div>;
}

function WorkspacePlaceholder({ workspace, onImport }: { workspace: WorkspaceId; onImport: () => void }) {
  const module = modules.find((item) => item.id === workspace);
  const Icon = module?.icon ?? Layers3;
  return <div className="workspace-view"><div className="workspace-heading"><div><span>{module?.eyebrow ?? "APPLICATION"}</span><h1>{module?.title ?? "任务中心"}</h1><p>{module?.description ?? "公共任务状态将在这里展示。"}</p></div></div><div className="large-empty"><Icon size={44} /><h2>模块骨架已经接入</h2><p>下一步将按独立规划接入真实输入、任务和模型流程。</p>{module && <button className="secondary-button" onClick={onImport}><FolderOpen size={18} />选择测试图片</button>}</div></div>;
}

function Onboarding({ onComplete }: { onComplete: () => void }) {
  const [step, setStep] = useState(0);
  const steps = [
    { icon: Aperture, eyebrow: "WELCOME", title: "你的本地视觉工作台", copy: "管理照片、生成穿搭预览，并逐步接入植物与实验室分析。核心资料默认留在这台电脑。" },
    { icon: UserRound, eyebrow: "ACCOUNT", title: "正式版从账户开始", copy: "账户和同步元数据使用云端主数据库；设备照片默认留在本机，只有你开启同步后才上传。" },
    { icon: Cpu, eyebrow: "ON DEMAND", title: "模型需要时再安装", copy: "首次启动不会下载几 GB 权重。没有模型时，相册和基础管理依然可以使用。" },
  ];
  const item = steps[step];
  const Icon = item.icon;
  return <div className="onboarding-backdrop"><section className="onboarding-card"><div className="onboarding-visual"><span>VISION / 01</span><Aperture size={84} /><small>PRIVATE · LOCAL · EXPLAINABLE</small></div><div className="onboarding-copy"><div className="step-dots">{steps.map((_, index) => <span key={index} className={step === index ? "is-active" : ""} />)}</div><div className="onboarding-icon"><Icon size={24} /></div><span className="eyebrow">{item.eyebrow}</span><h1>{item.title}</h1><p>{item.copy}</p><div className="onboarding-actions">{step > 0 && <button className="ghost-button" onClick={() => setStep(step - 1)}>返回</button>}<button className="primary-button" onClick={() => step === steps.length - 1 ? onComplete() : setStep(step + 1)}>{step === steps.length - 1 ? "进入开发预览" : "继续"}<ChevronRight size={17} /></button></div></div></section></div>;
}

function AccountDialog({ session, onAuthenticated, onClose }: { session: CloudSession | null; onAuthenticated: (session: CloudSession | null) => void; onClose: () => void }) {
  const [mode, setMode] = useState<"login" | "register">("login");
  const [email, setEmail] = useState("");
  const [displayName, setDisplayName] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const submit = async () => {
    setMessage(null);
    if (mode === "register" && password !== confirmPassword) {
      setMessage("两次输入的密码不一致");
      return;
    }
    if (mode === "register" && password.length < 15) {
      setMessage("密码至少需要 15 个字符");
      return;
    }
    setSubmitting(true);
    try {
      if (mode === "register") {
        await registerAccount(email, displayName, password);
        setMode("login");
        setPassword("");
        setConfirmPassword("");
        setMessage("注册请求已接受。开发服务开启自动验证时可直接登录；正式环境需要先验证邮箱。");
      } else {
        onAuthenticated(await loginAccount(email, password));
        onClose();
      }
    } catch (error) {
      setMessage(error instanceof Error ? error.message : String(error));
    } finally {
      setSubmitting(false);
    }
  };
  const logout = async () => {
    if (!session) return;
    setSubmitting(true);
    try {
      await logoutAccount(session);
    } catch {
      // Clear the local in-memory session even if the short-lived server is unavailable.
    } finally {
      onAuthenticated(null);
      setSubmitting(false);
      onClose();
    }
  };
  if (session) return <div className="dialog-backdrop" onMouseDown={onClose}><section className="dialog auth-dialog" onMouseDown={(event) => event.stopPropagation()}>
    <button className="dialog-close" onClick={onClose}><X size={18} /></button><div className="dialog-icon"><ShieldCheck /></div><span className="eyebrow">ACCOUNT</span><h2>{session.display_name}</h2><p>当前登录只保留在应用内存中，关闭应用后需要重新登录。安全的系统凭据持久化将在下一阶段接入。</p>
    <div className="account-state"><ShieldCheck size={18} /><div><strong>已登录</strong><span>访问令牌最长 15 分钟，服务器保存的令牌均为哈希</span></div></div>
    <button className="primary-button auth-submit" onClick={logout} disabled={submitting}>{submitting ? <LoaderCircle className="spin" size={17} /> : <LogIn size={17} />}退出当前设备</button>
  </section></div>;
  return <div className="dialog-backdrop" onMouseDown={onClose}><section className="dialog auth-dialog" onMouseDown={(event) => event.stopPropagation()}>
    <button className="dialog-close" onClick={onClose}><X size={18} /></button><div className="dialog-icon"><UserRound /></div><span className="eyebrow">ACCOUNT</span><h2>{mode === "login" ? "登录账户" : "创建账户"}</h2><p>连接账户服务器完成真实认证。设备照片和本机路径不会随登录请求发送。</p>
    <div className="auth-tabs"><button className={mode === "login" ? "is-active" : ""} onClick={() => setMode("login")}>登录</button><button className={mode === "register" ? "is-active" : ""} onClick={() => setMode("register")}>注册</button></div>
    <label className="auth-field"><span>邮箱</span><input type="email" autoComplete="email" placeholder="name@example.com" value={email} onChange={(event) => setEmail(event.target.value)} /></label>
    {mode === "register" && <label className="auth-field"><span>显示名称</span><input type="text" autoComplete="name" placeholder="用于应用内展示" maxLength={80} value={displayName} onChange={(event) => setDisplayName(event.target.value)} /></label>}
    <label className="auth-field"><span>密码</span><input type="password" autoComplete={mode === "login" ? "current-password" : "new-password"} placeholder={mode === "register" ? "至少 15 个字符" : "输入密码"} value={password} onChange={(event) => setPassword(event.target.value)} /></label>
    {mode === "register" && <label className="auth-field"><span>确认密码</span><input type="password" autoComplete="new-password" placeholder="再次输入密码" value={confirmPassword} onChange={(event) => setConfirmPassword(event.target.value)} /></label>}
    {message && <div className="auth-message"><CircleAlert size={15} /><span>{message}</span></div>}
    <button className="primary-button auth-submit" onClick={submit} disabled={submitting || !email || !password || (mode === "register" && !displayName)}>{submitting ? <LoaderCircle className="spin" size={17} /> : <LogIn size={17} />}{mode === "login" ? "登录" : "创建账户"}</button><button className="text-button" onClick={onClose}>暂不登录</button>
  </section></div>;
}

export default App;
