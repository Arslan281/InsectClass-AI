import TrainingAnalysis from "./TrainingAnalysis.jsx";
import {
    Activity,
    BrainCircuit,
    Database,
    GitCompareArrows,
    Layers3,
    Microscope
} from "lucide-react";

import {
    ResponsiveContainer,
    BarChart,
    Bar,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    Legend
} from "recharts";

const performance = [
    { model: "Original Single", accuracy: 22.51, precision: 22.36, recall: 21.36, f1: 18.45 },
    { model: "Improved Single", accuracy: 31.44, precision: 26.64, recall: 27.14, f1: 26.45 },
    { model: "Original Multi", accuracy: 21.63, precision: 21.57, recall: 25.49, f1: 20.27 },
    { model: "Improved Multi", accuracy: 43.54, precision: 42.68, recall: 38.78, f1: 39.86 }
];

const validationHistory = [
    { model: "Original Single", accuracy: 22.85, f1: 19.53 },
    { model: "Improved Single", accuracy: 30.18, f1: 25.76 },
    { model: "Original Multi", accuracy: 22.43, f1: 21.31 },
    { model: "Multi · 4,500", accuracy: 24.46, f1: 23.34 },
    { model: "Multi · 15,662 + aug", accuracy: 27.42, f1: 26.22 },
    { model: "Multi · 256/128", accuracy: 42.25, f1: 37.09 },
    { model: "Improved Multi", accuracy: 43.83, f1: 40.35 }
];

const tooltipStyle = {
    backgroundColor: "#15231b",
    border: "1px solid #31563c",
    borderRadius: "10px",
    color: "#e7f5e9"
};

export default function DeveloperStudio() {
    return (
        <section className="secondary-page studio-page">
            <span className="eyebrow">NEURAL NETWORK RESEARCH</span>
            <h1>Developer Studio</h1>
            <p>
                Explore our feedforward neural networks, training experiments,
                and evaluation results from the IP102 insect dataset.
            </p>

            <div className="metric-grid">
                <div className="metric-card">
                    <span>Single-Layer Test Accuracy</span>
                    <strong>31.44%</strong>
                    <small>7,841 unseen test images</small>
                </div>

                <div className="metric-card">
                    <span>Multilayer Test Macro F1</span>
                    <strong>39.86%</strong>
                    <small>Average across 15 categories</small>
                </div>

                <div className="metric-card">
                    <span>Selected ANN Models</span>
                    <strong>02</strong>
                    <small>Improved Single-Layer and Multilayer</small>
                </div>
            </div>

            <div className="studio-section-heading">
                <div>
                    <span>01 / FINAL EVALUATION</span>
                    <h2>Test Performance</h2>
                    <p>
                        Both improved networks and their original baselines were evaluated on the same
                        held-out test dataset containing 7,841 images.
                    </p>
                </div>
            </div>

            <div className="studio-chart-panel">
                <div className="studio-panel-title">
                    <GitCompareArrows size={20} />
                    <h3>Original vs Improved — Final Test Performance</h3>
                </div>

                <div className="studio-chart">
                    <ResponsiveContainer width="100%" height={310}>
                        <BarChart data={performance} barGap={6}>
                            <CartesianGrid
                                stroke="#263b2e"
                                strokeDasharray="3 3"
                                vertical={false}
                            />

                            <XAxis
                                dataKey="model"
                                stroke="#88a994"
                                tickLine={false}
                                axisLine={false}
                            />

                            <YAxis
                                domain={[0, 50]}
                                stroke="#88a994"
                                tickLine={false}
                                axisLine={false}
                                unit="%"
                            />

                            <Tooltip
                                contentStyle={tooltipStyle}
                                formatter={(value) => `${value.toFixed(2)}%`}
                            />

                            <Legend />

                            <Bar
                                dataKey="accuracy"
                                name="Accuracy"
                                fill="#7ee7a6"
                                radius={[5, 5, 0, 0]}
                            />

                            <Bar
                                dataKey="f1"
                                name="Macro F1"
                                fill="#46a5a0"
                                radius={[5, 5, 0, 0]}
                            />
                        </BarChart>
                    </ResponsiveContainer>
                </div>
            </div>

            <div className="studio-section-heading">
                <div>
                    <span>02 / EXPERIMENT HISTORY</span>
                    <h2>Validation Experiments</h2>
                    <p>
                        Selected validation milestones from original and improved configurations;
                        these scores were used for model selection before the final test.
                    </p>
                </div>
            </div>

            <div className="studio-chart-panel">
                <div className="studio-panel-title">
                    <Activity size={20} />
                    <h3>Validation Accuracy and Macro F1</h3>
                </div>

                <div className="studio-chart">
                    <ResponsiveContainer width="100%" height={320}>
                        <BarChart data={validationHistory}>
                            <CartesianGrid
                                stroke="#263b2e"
                                strokeDasharray="3 3"
                                vertical={false}
                            />

                            <XAxis
                                dataKey="model"
                                stroke="#88a994"
                                tick={{ fontSize: 11 }}
                                tickLine={false}
                                axisLine={false}
                            />

                            <YAxis
                                domain={[0, 50]}
                                stroke="#88a994"
                                tickLine={false}
                                axisLine={false}
                                unit="%"
                            />

                            <Tooltip
                                contentStyle={tooltipStyle}
                                formatter={(value) => `${value.toFixed(2)}%`}
                            />

                            <Legend />

                            <Bar
                                dataKey="accuracy"
                                name="Accuracy"
                                fill="#7ee7a6"
                                radius={[5, 5, 0, 0]}
                            />

                            <Bar
                                dataKey="f1"
                                name="Macro F1"
                                fill="#46a5a0"
                                radius={[5, 5, 0, 0]}
                            />
                        </BarChart>
                    </ResponsiveContainer>
                </div>
            </div>

            <div className="studio-section-heading">
                <div>
                    <span>03 / ARCHITECTURE</span>
                    <h2>Neural Network Architecture</h2>
                    <p>Understand the structure of our two selected models.</p>
                </div>
            </div>

            <div className="architecture-grid">
                <div className="architecture-card">
                    <div className="architecture-icon">
                        <Layers3 size={25} />
                    </div>

                    <h3>Single-Layer ANN</h3>
                    <p>
                        A feedforward neural network that maps flattened
                        image pixels directly to 15 output classes.
                    </p>

                    <div className="architecture-flow">
                        <div>Input: 64 × 64 × 3</div>
                        <span>↓</span>
                        <div>Channel normalization → Flatten: 12,288 features</div>
                        <span>↓</span>
                        <div>Dense(15): Softmax</div>
                    </div>

                    <div className="architecture-footer">
                        <span>Trainable Parameters</span>
                        <strong>184,335</strong>
                    </div>
                </div>

                <div className="architecture-card">
                    <div className="architecture-icon">
                        <BrainCircuit size={25} />
                    </div>

                    <h3>Improved Multilayer ANN</h3>
                    <p>
                        A multilayer feedforward network with two hidden
                        layers and ReLU activation functions.
                    </p>

                    <div className="architecture-flow">
                        <div>Input: 64 × 64 × 3</div>
                        <span>↓</span>
                        <div>Channel normalization → Resize: 32 × 32</div>
                        <span>↓</span>
                        <div>Flatten: 3,072 features</div>
                        <span>↓</span>
                        <div>Dense(512) → BatchNorm → ReLU → Dropout(30%)</div>
                        <span>↓</span>
                        <div>Dense(256) → BatchNorm → ReLU → Dropout(30%)</div>
                        <span>↓</span>
                        <div>Dense(15): Softmax</div>
                    </div>

                    <div className="architecture-footer">
                        <span>Trainable Parameters</span>
                        <strong>1,710,095</strong>
                    </div>
                </div>
            </div>

            <div className="studio-research-note">
                <Microscope size={21} />
                <div>
                    <h3>Research Observation</h3>
                    <p>
                        The improved multilayer ANN achieved 43.54% test accuracy
                        and 39.86% macro F1, compared with 31.44% accuracy and
                        26.45% macro F1 for the improved single-layer ANN.
                        Both are feedforward neural networks, not CNNs.
                        Their performance remains limited on difficult photographs;
                        predictions should not be treated as confirmed identification.
                    </p>
                </div>
            </div>

            <div className="studio-dataset-note">
                <Database size={17} />
                15 IP102 categories · 15,662 training images · 2,608 validation images · 7,841 test images.
            </div>
            <TrainingAnalysis />
        </section>
    );
}
