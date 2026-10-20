import InsectLibrary from "./components/InsectLibrary.jsx";
import DeveloperStudio from "./components/DeveloperStudio";
import { useEffect, useState } from "react";
import * as insectData from "./data/insects.js";
import {
  Activity, ArrowRight, BrainCircuit, Bug, CheckCircle2,
  ChevronRight, FlaskConical, ImagePlus, Layers3,
  Leaf, LoaderCircle, Menu, ScanSearch, UploadCloud,
  X, AlertCircle, BookOpen, Database, BarChart3, ShieldAlert, GitCompareArrows, Microscope, Cpu, Sparkles
} from "lucide-react";
import "./App.css";

const API_URL = "http://127.0.0.1:8000";

const navigation = [
  { id: "explorer", label: "Insect Explorer", icon: ScanSearch },
  { id: "studio", label: "Developer Studio", icon: BrainCircuit },
  { id: "library", label: "Insect Library", icon: Bug },
  { id: "about", label: "About Project", icon: FlaskConical },
];

const insectNames = [
  "Lycorma delicatula", "Aphids", "Blister beetle",
  "Corn borer", "Mole cricket", "Beet army worm",
  "Legume blister beetle", "Limacodidae", "Locustoidea",
  "Cicadella viridis", "Xylotrechus", "Grub",
  "Rice water weevil", "Flea beetle",
  "Icerya purchasi Maskell"
];

// Works with either a named or default export from insects.js.
const insectProfiles = insectData.insects ?? insectData.default ?? [];

function InsectFactToast({ insect, onClose, onOpenProfile }) {
  if (!insect) return null;

  return (
    <aside className="insect-fact-toast" role="status" aria-label="Insect fun fact">
      <div className="insect-fact-top">
        <span><Sparkles size={16} /> DID YOU KNOW?</span>
        <button type="button" className="insect-fact-close" onClick={onClose} aria-label="Close fun fact">
          <X size={17} />
        </button>
      </div>
      <div className="insect-fact-body">
        <img src={insect.image} alt={insect.displayName} />
        <div>
          <span className="insect-fact-eyebrow">SINGLE-LAYER ANN PREDICTION</span>
          <h3>{insect.displayName}</h3>
          <p>{insect.fact}</p>
        </div>
      </div>
      <button type="button" className="insect-fact-link" onClick={() => onOpenProfile(insect.name)}>
        Explore this insect <ArrowRight size={16} />
      </button>
      <small>Educational fact about a predicted category, not a confirmed identification.</small>
      <div className="insect-fact-timer" />
    </aside>
  );
}

function PredictionCard({ title, subtitle, data, onViewProfile }) {
  return (
    <div className="prediction-card">
      <div className="prediction-heading">
        <div className="model-symbol">
          <BrainCircuit size={20} />
        </div>
        <div>
          <h3>{title}</h3>
          <p>{subtitle}</p>
        </div>
      </div>

      <div className="predicted-label">
        <span>TOP PREDICTION</span>
        <h2>{data.predicted_insect}</h2>
      </div>

      <div className="prediction-list">
        {data.top_predictions.map((item, index) => (
          <div className="prediction-row" key={item.insect}>
            <div className="prediction-meta">
              <span>{index + 1}. {item.insect}</span>
              <strong>{item.confidence.toFixed(2)}%</strong>
            </div>
            <div className="progress-track">
              <div
                className="progress-fill"
                style={{ width: `${item.confidence}%` }}
              />
            </div>
          </div>
        ))}
      </div>
      <button
        className="view-profile-button"
        onClick={() => onViewProfile(data.predicted_insect)}
      >
        <BookOpen size={17} />
        View Insect Profile
        <ArrowRight size={17} />
      </button>
    </div>
  );
}

export default function App() {
  const [page, setPage] = useState("explorer");
  const [selectedInsectName, setSelectedInsectName] = useState(null);
  const [factToast, setFactToast] = useState(null);

  useEffect(() => {
    if (!factToast) return;
    const timeout = setTimeout(() => setFactToast(null), 8000);
    return () => clearTimeout(timeout);
  }, [factToast]);

  function openInsectProfile(insectName) {
    setFactToast(null);
    setSelectedInsectName(insectName);
    setPage("library");
    window.scrollTo(0, 0);
  }
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState("");
  const [result, setResult] = useState(null);
  const [rejection, setRejection] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [mobileMenu, setMobileMenu] = useState(false);

  function handleFile(selectedFile) {
    if (!selectedFile) return;

    if (!["image/jpeg", "image/png", "image/webp"].includes(selectedFile.type)) {
      setError("Please select a JPG, PNG, or WebP image.");
      return;
    }

    if (selectedFile.size > 10 * 1024 * 1024) {
      setError("Image size must be 10 MB or smaller.");
      return;
    }

    if (preview) URL.revokeObjectURL(preview);

    setFile(selectedFile);
    setPreview(URL.createObjectURL(selectedFile));
    setResult(null);
    setRejection(null);
    setFactToast(null);
    setError("");
  }

  async function analyzeImage() {
    if (!file || loading) return;

    setLoading(true);
    setError("");
    setResult(null);
    setRejection(null);
    setFactToast(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Prediction failed.");
      }

      // A rejected image is a valid API result, not a server error.
      // The binary gate must pass before either ANN prediction is displayed.
      if (data.is_insect === false) {
        setRejection(data);
        setResult(null);
        return;
      }

      if (!data.predictions?.single_layer || !data.predictions?.multilayer) {
        throw new Error("The AI server returned incomplete prediction results.");
      }

      setRejection(null);
      setResult(data.predictions);

      const predictedName = data.predictions?.single_layer?.predicted_insect;
      const classId = insectNames.indexOf(predictedName);
      const profile = insectProfiles.find((entry) => Number(entry.id) === classId);
      if (profile?.fact && classId !== -1) {
        setFactToast({
          name: predictedName,
          displayName: profile.commonName || profile.name || predictedName,
          fact: profile.fact,
          image: `/insects/insect_${String(classId).padStart(2, "0")}_1.jpg`,
          shownAt: Date.now()
        });
      }
    } catch (err) {
      setError(
        err.message === "Failed to fetch"
          ? "Cannot reach the AI server. Make sure FastAPI is running on port 8000."
          : err.message
      );
    } finally {
      setLoading(false);
    }
  }

  function clearImage() {
    if (preview) URL.revokeObjectURL(preview);
    setFile(null);
    setPreview("");
    setResult(null);
    setRejection(null);
    setFactToast(null);
    setError("");
  }

  const currentPage = navigation.find((item) => item.id === page);

  return (
    <div className="app-shell">
      <aside className={`sidebar ${mobileMenu ? "sidebar-open" : ""}`}>
        <div className="brand">
          <div className="brand-icon"><Bug size={25} /></div>
          <div>
            <strong>INSECTCLASS<span> AI</span></strong>
            <small>INTELLIGENT RECOGNITION</small>
          </div>
        </div>

        <div className="sidebar-section-title">WORKSPACE</div>

        <nav className="sidebar-nav">
          {navigation.map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              className={`nav-item ${page === id ? "active" : ""}`}
              onClick={() => {
                setPage(id);
                setFactToast(null);
                if (id === "library") setSelectedInsectName(null);
                setMobileMenu(false);
              }}
            >
              <Icon size={19} />
              <span>{label}</span>
              {page === id && <ChevronRight size={16} />}
            </button>
          ))}
        </nav>

        <div className="sidebar-bottom">
          <div className="status-card">
            <div className="status-header">
              <span className="status-dot" />
              <strong>AI Engine</strong>
            </div>
            <p>Single-Layer & Multilayer ANN</p>
            <span>15 insect categories</span>
          </div>
          <p className="sidebar-footer">INSECTCLASS AI • UNIVERSITY PROJECT</p>
        </div>
      </aside>

      {mobileMenu && (
        <button
          className="mobile-overlay"
          onClick={() => setMobileMenu(false)}
          aria-label="Close navigation"
        />
      )}

      <main className="main-content">
        <header className="topbar">
          <div className="topbar-left">
            <button
              className="mobile-menu-button"
              onClick={() => setMobileMenu(!mobileMenu)}
              aria-label="Open navigation"
            >
              <Menu size={22} />
            </button>
            <span>Workspace</span>
            <ChevronRight size={15} />
            <strong>{currentPage.label}</strong>
          </div>
          <div className="topbar-right">
            <span className="version-tag">v1.0 • ANN ENGINE</span>
            <div className="avatar">IC</div>
          </div>
        </header>

        <div className="page-container">
          {page === "explorer" && (
            <>
              <section className="hero">
                <div className="hero-content">
                  <span className="eyebrow">
                    <span className="eyebrow-dot" />
                    ARTIFICIAL NEURAL NETWORKS
                  </span>
                  <h1>
                    Discover the world of
                    <br />
                    <span>intelligent insect recognition.</span>
                  </h1>
                  <p>
                    Upload an insect photograph and explore predictions
                    from two feedforward neural networks trained on
                    agricultural insect imagery.
                  </p>
                  <div className="hero-meta">
                    <span><Layers3 size={16} /> 15 Categories</span>
                    <span><BrainCircuit size={16} /> 2 ANN Models</span>
                    <span><Activity size={16} /> Live Inference</span>
                  </div>
                </div>
                <div className="hero-decoration">
                  <div className="hero-glow" />
                  <Bug size={140} strokeWidth={0.8} />
                </div>
              </section>

              <section className="section-heading">
                <div>
                  <div className="section-kicker">01 / IDENTIFICATION</div>
                  <h2>Insect Explorer</h2>
                  <p>Analyze an image using our trained ANN classifiers.</p>
                </div>
              </section>

              <div className="explorer-grid">
                <div className="panel upload-panel">
                  <div className="panel-heading">
                    <div className="panel-title">
                      <ImagePlus size={19} />
                      <h3>Image Input</h3>
                    </div>
                    <span className="mini-tag">JPG / PNG / WEBP</span>
                  </div>

                  {!preview ? (
                    <label
                      className="upload-zone"
                      onDragOver={(e) => e.preventDefault()}
                      onDrop={(e) => {
                        e.preventDefault();
                        handleFile(e.dataTransfer.files[0]);
                      }}
                    >
                      <input
                        type="file"
                        accept="image/jpeg,image/png,image/webp"
                        onChange={(e) => handleFile(e.target.files[0])}
                        hidden
                      />
                      <div className="upload-icon">
                        <UploadCloud size={30} />
                      </div>
                      <h3>Drop an insect image here</h3>
                      <p>or click to browse your files</p>
                      <span className="upload-hint">Maximum file size: 10 MB</span>
                    </label>
                  ) : (
                    <div className="image-preview-wrapper">
                      <img src={preview} alt="Uploaded insect" />
                      <button
                        className="remove-image"
                        onClick={clearImage}
                        title="Remove image"
                      >
                        <X size={18} />
                      </button>
                      <div className="preview-caption">
                        <CheckCircle2 size={16} />
                        <span>{file.name}</span>
                      </div>
                    </div>
                  )}

                  {error && (
                    <div className="error-message">
                      <AlertCircle size={17} />
                      {error}
                    </div>
                  )}

                  {rejection && (
                    <div
                      role="alert"
                      aria-live="assertive"
                      style={{
                        marginTop: "16px",
                        padding: "18px",
                        borderRadius: "12px",
                        border: "1px solid rgba(245, 158, 11, 0.55)",
                        background: "rgba(245, 158, 11, 0.11)",
                        color: "#f6d394",
                      }}
                    >
                      <div style={{ display: "flex", alignItems: "center", gap: "9px", marginBottom: "8px" }}>
                        <AlertCircle size={22} />
                        <strong style={{ fontSize: "17px" }}>No insect detected</strong>
                      </div>
                      <p style={{ margin: 0, lineHeight: 1.6 }}>
                        {rejection.message || "Please upload a photograph containing an insect."}
                      </p>
                      <small style={{ display: "block", marginTop: "8px", opacity: 0.8 }}>
                        The ANN classifiers were skipped. The screening model can make mistakes.
                      </small>
                    </div>
                  )}

                  {result && (
                    <div role="status" style={{ marginTop: "14px", color: "#8befbc", display: "flex", gap: "8px", alignItems: "center" }}>
                      <CheckCircle2 size={18} />
                      Insect detected. Classification results are below.
                    </div>
                  )}

                  <button
                    className="primary-button"
                    disabled={!file || loading}
                    onClick={analyzeImage}
                  >
                    {loading ? (
                      <>
                        <LoaderCircle className="spin" size={19} />
                        Analyzing image...
                      </>
                    ) : (
                      <>
                        <ScanSearch size={19} />
                        Analyze Insect
                        <ArrowRight size={17} />
                      </>
                    )}
                  </button>
                </div>

                <div className="panel insight-panel">
                  <div className="panel-heading">
                    <div className="panel-title">
                      <Activity size={19} />
                      <h3>Analysis Overview</h3>
                    </div>
                    <span className="mini-tag">MODEL OUTPUT</span>
                  </div>

                  <div className="overview-stat">
                    <span>Classification Models</span>
                    <strong>02</strong>
                  </div>
                  <div className="overview-stat">
                    <span>Recognizable Categories</span>
                    <strong>15</strong>
                  </div>
                  <div className="overview-stat">
                    <span>Input Resolution</span>
                    <strong>64 × 64</strong>
                  </div>

                  <div className="info-box">
                    <Leaf size={19} />
                    <p>
                      This is an experimental university research system.
                      Predictions are not guaranteed to be correct and
                      should not be used for pest-control decisions.
                    </p>
                  </div>
                </div>
              </div>

              {result && (
                <section className="results-section">
                  <div className="section-heading">
                    <div>
                      <div className="section-kicker">02 / INFERENCE OUTPUT</div>
                      <h2>Classification Results</h2>
                      <p>Side-by-side predictions from both ANN architectures.</p>
                    </div>
                  </div>

                  <div className="result-grid">
                    <PredictionCard
                      title="Single-Layer ANN"
                      subtitle="Flatten → Softmax"
                      data={result.single_layer}
                      onViewProfile={openInsectProfile}
                    />
                    <PredictionCard
                      title="Multilayer ANN"
                      subtitle="Dense(512) → Dense(256) → Softmax"
                      data={result.multilayer}
                      onViewProfile={openInsectProfile}
                    />
                  </div>
                  <p className="results-disclaimer">
                    Scores represent model-generated probabilities, not
                    calibrated confidence or confirmed insect identities.
                  </p>
                </section>
              )}
            </>
          )}

          {page === "studio" && <DeveloperStudio />}

          {page === "library" && (
            <InsectLibrary
              selectedInsectName={selectedInsectName}
              onExplore={() => setPage("explorer")}
            />
          )}

          {page === "about" && (
            <section className="secondary-page about-research">
              <div className="about-hero">
                <div className="about-hero-copy">
                  <span className="eyebrow"><span className="eyebrow-dot" /> RESEARCH PROJECT / ARTIFICIAL NEURAL NETWORKS</span>
                  <h1>Inside <span>InsectClass AI.</span></h1>
                  <p>
                    A university research project exploring how single-layer and multilayer
                    feedforward neural networks classify agricultural insect photographs.
                    Built for transparent experimentation, comparison, and learning.
                  </p>
                  <div className="about-hero-tags">
                    <span><Database size={15} /> IP102 dataset</span>
                    <span><Bug size={15} /> 15 selected classes</span>
                    <span><BrainCircuit size={15} /> 2 finalist ANNs</span>
                  </div>
                </div>
                <div className="about-hero-icon"><BrainCircuit size={112} strokeWidth={0.9} /></div>
              </div>

              <div className="about-stat-grid">
                <div className="about-stat"><span>Finalist training set</span><strong>15,662</strong><small>Official IP102 training split for 15 classes</small></div>
                <div className="about-stat"><span>Validation set</span><strong>2,608</strong><small>Used for model selection</small></div>
                <div className="about-stat"><span>Untouched test set</span><strong>7,841</strong><small>Final evaluation only</small></div>
                <div className="about-stat"><span>Classes recognized</span><strong>15</strong><small>Selected from IP102</small></div>
              </div>

              <div className="about-section-title">
                <span>01 / DATA PIPELINE</span>
                <h2><Database size={23} /> Dataset & preprocessing</h2>
                <p>The project uses a subset of IP102, an agricultural pest image dataset. The official split files were used to separate the selected classes.</p>
              </div>
              <div className="about-grid">
                <article className="panel about-detail-card">
                  <div className="about-card-icon"><Layers3 size={22} /></div>
                  <h3>Data preparation</h3>
                  <p>Each image is converted to RGB, resized to 64 × 64 pixels, and normalized to floating-point values between 0 and 1. Class labels are encoded across 15 categories.</p>
                  <div className="about-pill-row"><span>RGB</span><span>64 × 64</span><span>0–1 scaling</span></div>
                </article>
                <article className="panel about-detail-card">
                  <div className="about-card-icon"><Microscope size={22} /></div>
                  <h3>Experimental data expansion</h3>
                  <p>The final improved ANN models were trained on 15,662 official training images across 15 classes. Early experiments used a balanced subset of 4,500 images.</p>
                  <div className="about-pill-row"><span>15,662 final training images</span><span>Class counts vary</span></div>
                </article>
              </div>

              <div className="about-section-title">
                <span>02 / MODEL DESIGN</span>
                <h2><Cpu size={23} /> Our neural networks</h2>
                <p>Two feedforward ANN architectures were selected for comparison. Neither is a convolutional neural network or a pretrained image model.</p>
              </div>
              <div className="about-grid">
                <article className="panel about-detail-card">
                  <div className="about-card-icon"><Layers3 size={22} /></div>
                  <h3>Single-Layer ANN</h3>
                  <div className="about-model-path">64 × 64 RGB → Flatten → 15-class Softmax</div>
                  <p>A simple baseline without hidden layers. Flatten converts each image into a feature vector, and softmax produces class probability scores.</p>
                  <strong className="about-parameter">184,335 trainable parameters</strong>
                </article>
                <article className="panel about-detail-card">
                  <div className="about-card-icon"><BrainCircuit size={22} /></div>
                  <h3>Improved Multilayer ANN</h3>
                  <div className="about-model-path">Resize to 32 × 32 → Flatten → 512 ReLU → 256 ReLU → Softmax</div>
                  <p>Two hidden dense layers with batch normalization and 30% dropout learn nonlinear combinations of pixel features. The model resizes images to 32 × 32 internally.</p>
                  <strong className="about-parameter">1,710,095 trainable parameters</strong>
                </article>
              </div>

              <div className="about-section-title">
                <span>03 / FINAL EVALUATION</span>
                <h2><BarChart3 size={23} /> Measured test performance</h2>
                <p>Results from a single final evaluation on 7,841 held-out test images, after selecting the finalist models using validation results.</p>
              </div>
              <div className="panel about-evaluation">
                <div className="about-eval-title"><GitCompareArrows size={20} /><h3>Finalist comparison</h3><span>TEST SET · 7,841 IMAGES</span></div>
                <div className="about-table-scroll">
                  <table className="about-results-table">
                    <thead><tr><th>Metric</th><th>Single-Layer ANN</th><th>Improved Multilayer ANN</th></tr></thead>
                    <tbody>
                      <tr><td>Accuracy</td><td>31.44%</td><td>43.54%</td></tr>
                      <tr><td>Macro precision</td><td>26.64%</td><td>42.68%</td></tr>
                      <tr><td>Macro recall</td><td>27.14%</td><td>38.78%</td></tr>
                      <tr><td>Macro F1-score</td><td>26.45%</td><td>39.86%</td></tr>
                    </tbody>
                  </table>
                </div>
                <p className="about-eval-note">The improved multilayer model outperformed the improved single-layer model on accuracy and macro F1. Both substantially improved over the original baselines, but neither is dependable enough for confirmed species identification.</p>
                <button type="button" className="about-studio-link" onClick={() => { setPage("studio"); window.scrollTo(0, 0); }}>
                  <BarChart3 size={17} /> Explore training graphs & confusion matrices <ArrowRight size={17} />
                </button>
              </div>

              <div className="about-section-title">
                <span>04 / SCOPE & NEXT STEPS</span>
                <h2><ShieldAlert size={23} /> Limitations & future work</h2>
              </div>
              <div className="about-grid">
                <article className="panel about-detail-card">
                  <div className="about-card-icon"><AlertCircle size={22} /></div>
                  <h3>Current limitations</h3>
                  <p>Many categories are visually similar, some labels describe broad insect groups, and image quality and class frequencies vary. Flattened-pixel ANNs also lack the spatial feature extraction of CNNs. Scores are not calibrated certainty.</p>
                </article>
                <article className="panel about-detail-card">
                  <div className="about-card-icon"><Leaf size={22} /></div>
                  <h3>Possible improvements</h3>
                  <p>Curate labels and photos, balance class representation, explore augmentation, evaluate on new photographs, calibrate probability scores, and compare a convolutional model in future work.</p>
                </article>
              </div>
              <div className="about-caution"><ShieldAlert size={21} /><p><strong>Research demonstration only.</strong> InsectClass AI is an educational prototype. Its predictions must not be treated as confirmed species identifications or used alone for agricultural pest-management decisions.</p></div>
            </section>
          )}
        </div>
      </main>
      <InsectFactToast
        key={factToast?.shownAt ?? "empty"}
        insect={factToast}
        onClose={() => setFactToast(null)}
        onOpenProfile={openInsectProfile}
      />
    </div>
  );
}
