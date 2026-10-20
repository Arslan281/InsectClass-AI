import { useEffect, useState } from "react";
import { Activity, BrainCircuit, GitCompareArrows, AlertCircle } from "lucide-react";
import {
    ResponsiveContainer, LineChart, Line, CartesianGrid,
    XAxis, YAxis, Tooltip, Legend
} from "recharts";

// Files are copied from D:\Claude_InsectClass\results to frontend/public/evaluation.
// These are the FINAL selected models; do not use the older v4 plots.
const histories = [
    {
        name: "Improved Single-Layer ANN",
        file: "/evaluation/improved_single_layer_s_full_noaug_history.json",
        bestEpoch: 22
    },
    {
        name: "Improved Multilayer ANN (512 / 256)",
        file: "/evaluation/improved_multilayer_m_final_512_history.json",
        bestEpoch: 43
    }
];

const confusionMatrices = [
    {
        name: "Improved Single-Layer ANN",
        file: "/evaluation/single_layer_improved_test_cm.png"
    },
    {
        name: "Improved Multilayer ANN (512 / 256)",
        file: "/evaluation/multilayer_improved_test_cm.png"
    }
];

const tooltipStyle = {
    backgroundColor: "#15231b",
    border: "1px solid #31563c",
    borderRadius: "10px",
    color: "#e7f5e9"
};

function HistoryChart({ data, metric, title }) {
    const isAccuracy = metric === "accuracy";
    return (
        <div className="studio-chart-panel">
            <div className="studio-panel-title">
                <Activity size={20} />
                <h3>{title}</h3>
            </div>
            <div className="studio-chart">
                <ResponsiveContainer width="100%" height={290}>
                    <LineChart data={data}>
                        <CartesianGrid stroke="#263b2e" strokeDasharray="3 3" vertical={false} />
                        <XAxis dataKey="epoch" stroke="#88a994" tickLine={false} axisLine={false} allowDecimals={false} />
                        <YAxis
                            stroke="#88a994" tickLine={false} axisLine={false}
                            tickFormatter={isAccuracy ? (v) => `${Math.round(v * 100)}%` : (v) => Number(v).toFixed(1)}
                        />
                        <Tooltip
                            contentStyle={tooltipStyle}
                            formatter={(value) => isAccuracy ? `${(Number(value) * 100).toFixed(2)}%` : Number(value).toFixed(4)}
                            labelFormatter={(epoch) => `Epoch ${epoch}`}
                        />
                        <Legend />
                        <Line type="monotone" dataKey={metric} name="Training" stroke="#7ee7a6" strokeWidth={3} dot={false} />
                        <Line type="monotone" dataKey={`val_${metric}`} name="Validation" stroke="#46a5a0" strokeWidth={3} dot={false} />
                    </LineChart>
                </ResponsiveContainer>
            </div>
        </div>
    );
}

function ConfusionMatrixCard({ name, file }) {
    const [missing, setMissing] = useState(false);
    return (
        <div className="confusion-card">
            <div className="studio-panel-title">
                <GitCompareArrows size={20} />
                <h3>{name}</h3>
            </div>
            {missing ? (
                <p role="status">The final test confusion matrix has not been copied into the website's evaluation folder yet.</p>
            ) : (
                <a href={file} target="_blank" rel="noopener noreferrer">
                    <img
                        src={file}
                        alt={`${name} final test confusion matrix`}
                        loading="lazy"
                        onError={() => setMissing(true)}
                    />
                </a>
            )}
            {!missing && <p>Click the matrix to view it at full resolution.</p>}
        </div>
    );
}

export default function TrainingAnalysis() {
    const [historyData, setHistoryData] = useState([]);
    const [loading, setLoading] = useState(true);
    const [errors, setErrors] = useState([]);

    useEffect(() => {
        let cancelled = false;
        async function loadHistories() {
            const results = await Promise.all(histories.map(async (item) => {
                try {
                    const response = await fetch(item.file);
                    if (!response.ok) throw new Error(`Missing ${item.file}`);
                    const history = await response.json();
                    if (![history.loss, history.accuracy, history.val_loss, history.val_accuracy].every(Array.isArray)) {
                        throw new Error(`Invalid training history format: ${item.file}`);
                    }
                    const rows = history.loss.map((_, i) => ({
                        epoch: i + 1,
                        accuracy: history.accuracy[i],
                        val_accuracy: history.val_accuracy[i],
                        loss: history.loss[i],
                        val_loss: history.val_loss[i]
                    }));
                    return { name: item.name, rows, bestEpoch: item.bestEpoch };
                } catch (e) {
                    return { name: item.name, error: e.message };
                }
            }));
            if (!cancelled) {
                setHistoryData(results.filter((r) => !r.error));
                setErrors(results.filter((r) => r.error).map((r) => r.error));
                setLoading(false);
            }
        }
        loadHistories();
        return () => { cancelled = true; };
    }, []);

    return (
        <>
            <div className="studio-section-heading">
                <div>
                    <span>04 / TRAINING HISTORY</span>
                    <h2>Accuracy &amp; Loss Curves</h2>
                    <p>Actual training and validation histories of the two selected improved ANN models.</p>
                </div>
            </div>

            {loading && <div className="studio-chart-panel">Loading training history...</div>}
            {errors.map((error) => (
                <div className="error-message" key={error} role="status">
                    <AlertCircle size={18} /> {error}. Copy the corresponding JSON from the experimental results folder.
                </div>
            ))}
            {historyData.map((model) => (
                <div className="training-model-section" key={model.name}>
                    <div className="training-model-heading">
                        <BrainCircuit size={20} />
                        <h3>{model.name}</h3>
                        <span>{model.rows.length} epochs • best checkpoint: epoch {model.bestEpoch}</span>
                    </div>
                    <div className="training-chart-grid">
                        <HistoryChart data={model.rows} metric="accuracy" title="Accuracy by Epoch" />
                        <HistoryChart data={model.rows} metric="loss" title="Loss by Epoch" />
                    </div>
                </div>
            ))}
            <p className="results-disclaimer">
                Early stopping restored the weights from the best epoch according to validation accuracy.
                The final point of each curve represents the last trained epoch, not necessarily the saved checkpoint.
            </p>

            <div className="studio-section-heading">
                <div>
                    <span>05 / CLASSIFICATION ANALYSIS</span>
                    <h2>Final Test Confusion Matrices</h2>
                    <p>Actual versus predicted categories across the 7,841 held-out test images.</p>
                </div>
            </div>
            <div className="confusion-grid">
                {confusionMatrices.map((item) => <ConfusionMatrixCard key={item.file} {...item} />)}
            </div>
        </>
    );
}
