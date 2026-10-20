
import { useEffect, useState } from "react";
import {
    ArrowLeft,
    ArrowRight,
    BookOpen,
    Bug,
    Leaf,
    Search,
    Sprout,
    MapPin,
    Lightbulb,
    Microscope,
    ChevronLeft,
    ChevronRight,
    Images,
    Globe2
} from "lucide-react";

import { insects } from "../data/insects";

export default function InsectLibrary({
    onExplore,
    selectedInsectName
}) {
    const [selected, setSelected] = useState(() =>
        insects.find(
            (insect) => insect.name === selectedInsectName
        ) || null
    );
    const [search, setSearch] = useState("");
    const [galleries, setGalleries] = useState({});
    const [activePhoto, setActivePhoto] = useState(0);
    const [enlarged, setEnlarged] = useState(false);

    useEffect(() => {
        fetch("/insects/galleries.json")
            .then((response) => {
                if (!response.ok) throw new Error("Gallery unavailable");
                return response.json();
            })
            .then(setGalleries)
            .catch((error) => console.error(error));
    }, []);

    const filtered = insects.filter((insect) =>
        `${insect.name} ${insect.commonName} ${insect.category}`
            .toLowerCase()
            .includes(search.toLowerCase())
    );

    if (selected) {
        return (
            <section className="insect-profile">
                <button
                    className="profile-back"
                    onClick={() => setSelected(null)}
                >
                    <ArrowLeft size={17} />
                    Back to Insect Library
                </button>

                <div className="profile-hero">
                    <div>
                        <span className="eyebrow">
                            INSECT PROFILE {String(selected.id + 1).padStart(2, "0")} / 15
                        </span>

                        <h1>{selected.commonName}</h1>
                        <p className="profile-scientific">{selected.name}</p>
                        <p className="profile-description">{selected.description}</p>
                    </div>

                    <div className="profile-hero-icon">
                        <Bug size={110} strokeWidth={1} />
                    </div>
                </div>

                {/* CLASSIFICATION, HABITAT, DISTRIBUTION AND DIET */}

                {galleries[selected.id]?.length > 0 && (
                    <div className="insect-photo-gallery">
                        <div className="gallery-main">
                            <img
                                src={galleries[selected.id][activePhoto]?.src}
                                alt={`${selected.commonName} photograph ${activePhoto + 1}`}
                                onClick={() => setEnlarged(true)}
                            />

                            <div className="gallery-counter">
                                <Images size={15} />
                                {activePhoto + 1} / {galleries[selected.id].length}
                            </div>
                            <div className="gallery-caption">
                                {galleries[selected.id][activePhoto]?.caption ||
                                    "Insect photograph"}
                            </div>

                            <button
                                className="gallery-arrow gallery-prev"
                                aria-label="Previous photograph"
                                onClick={() =>
                                    setActivePhoto((current) =>
                                        (current - 1 + galleries[selected.id].length) %
                                        galleries[selected.id].length
                                    )
                                }
                            >
                                <ChevronLeft size={22} />
                            </button>

                            <button
                                className="gallery-arrow gallery-next"
                                aria-label="Next photograph"
                                onClick={() =>
                                    setActivePhoto((current) =>
                                        (current + 1) % galleries[selected.id].length
                                    )
                                }
                            >
                                <ChevronRight size={22} />
                            </button>
                        </div>

                        <div className="gallery-thumbnails">
                            {galleries[selected.id].map((photo, index) => (
                                <button
                                    key={photo.src}
                                    className={`gallery-thumb ${activePhoto === index ? "selected" : ""
                                        }`}
                                    onClick={() => setActivePhoto(index)}
                                    aria-label={`View photograph ${index + 1}`}
                                >
                                    <img
                                        src={photo.src}
                                        alt={`${selected.commonName} thumbnail ${index + 1}`}
                                    />
                                </button>
                            ))}
                        </div>
                    </div>
                )}

                {enlarged && galleries[selected.id]?.[activePhoto] && (
                    <div
                        className="gallery-lightbox"
                        onClick={() => setEnlarged(false)}
                        role="presentation"
                    >
                        <button
                            className="gallery-close"
                            aria-label="Close enlarged photograph"
                            onClick={() => setEnlarged(false)}
                        >
                            ×
                        </button>

                        <img
                            src={galleries[selected.id][activePhoto].src}
                            alt={`Enlarged ${selected.commonName}`}
                        />
                    </div>
                )}


                <div className="profile-info-grid">
                    <div className="profile-info-card">
                        <Microscope size={20} />
                        <span>CLASSIFICATION</span>
                        <strong>{selected.classification}</strong>
                    </div>

                    <div className="profile-info-card">
                        <MapPin size={20} />
                        <span>HABITAT</span>
                        <strong>{selected.habitat}</strong>
                    </div>

                    <div className="profile-info-card">
                        <Globe2 size={20} />
                        <span>GEOGRAPHICAL DISTRIBUTION</span>
                        <strong>{selected.distribution || "Not yet available"}</strong>
                    </div>

                    <div className="profile-info-card">
                        <Leaf size={20} />
                        <span>DIET</span>
                        <strong>{selected.diet}</strong>
                    </div>
                </div>

                {/* INSECT INFORMATION */}

                {selected.lifeCycle && (
                    <div className="life-cycle-panel">
                        <div className="life-cycle-heading">
                            <h2>Insect Life Cycle</h2>
                            <span>{selected.lifeCycle.type}</span>
                        </div>

                        <div className="life-cycle-stages">
                            {selected.lifeCycle.stages.map((stage, index) => (
                                <div className="life-cycle-step" key={stage}>
                                    <div className="life-cycle-stage">
                                        <span>
                                            STAGE {String(index + 1).padStart(2, "0")}
                                        </span>
                                        <strong>{stage}</strong>
                                    </div>

                                    {index < selected.lifeCycle.stages.length - 1 && (
                                        <ArrowRight
                                            className="life-cycle-arrow"
                                            size={20}
                                        />
                                    )}
                                </div>
                            ))}
                        </div>

                        <p>{selected.lifeCycle.description}</p>
                    </div>
                )}


                <div className="profile-content-grid">
                    <div className="panel profile-text-panel">
                        <h2>
                            <BookOpen size={21} />
                            About This Insect
                        </h2>
                        <p>{selected.description}</p>

                        <h2>
                            <Sprout size={21} />
                            Agricultural Relevance
                        </h2>
                        <p>{selected.impact}</p>
                    </div>

                    <div className="panel profile-fact-panel">
                        <Lightbulb size={25} />

                        <h3>Did You Know?</h3>
                        <p>{selected.fact}</p>

                        <div className="profile-category">
                            <span>DATASET CATEGORY</span>
                            <strong>{selected.category}</strong>
                        </div>
                    </div>
                </div>

                {/* NAVIGATE TO INSECT EXPLORER */}

                <button
                    className="profile-explore-button"
                    onClick={onExplore}
                >
                    <span>
                        <strong>Try Insect Explorer</strong>
                        <small>
                            Upload an image and compare ANN predictions.
                        </small>
                    </span>

                    <ArrowRight size={23} />
                </button>
            </section>
        );
    }

    return (
        <section className="secondary-page insect-library-page">
            <span className="eyebrow">
                ENTOMOLOGY COLLECTION
            </span>

            <h1>Insect Library</h1>

            <p>
                Explore 15 agricultural insect categories from the IP102
                dataset. Select any insect to view its dedicated profile.
            </p>

            {/* SEARCH BAR */}

            <div className="library-toolbar">
                <Search size={19} />

                <input
                    type="text"
                    placeholder="Search insects, names, or categories..."
                    value={search}
                    onChange={(event) => setSearch(event.target.value)}
                />

                <span>{filtered.length} / 15</span>
            </div>

            {/* INSECT CARDS */}

            <div className="library-grid">
                {filtered.map((insect) => (
                    <button
                        className="library-card library-clickable"
                        key={insect.id}
                        onClick={() => {
                            setSelected(insect);
                            setSelected(insect);
                            setActivePhoto(0);
                            setEnlarged(false);
                            window.scrollTo(0, 0);
                        }}
                    >
                        {galleries[insect.id]?.[0] ? (
                            <div className="library-cover">
                                <img
                                    src={galleries[insect.id][0].src}
                                    alt={insect.commonName}
                                    loading="lazy"
                                />
                            </div>
                        ) : (
                            <div className="library-icon">
                                <Bug size={25} />
                            </div>
                        )}

                        <span>
                            CATEGORY {String(insect.id + 1).padStart(2, "0")}
                        </span>

                        <h3>{insect.commonName}</h3>

                        <p className="library-scientific">
                            {insect.name}
                        </p>

                        <div className="library-card-bottom">
                            <span>{insect.category}</span>
                            <ArrowRight size={17} />
                        </div>
                    </button>
                ))}
            </div>

            {/* EMPTY SEARCH RESULTS */}

            {filtered.length === 0 && (
                <div className="library-empty">
                    <Bug size={35} />
                    <h3>No insects found</h3>
                    <p>Try searching for a different name or category.</p>
                </div>
            )}
        </section>
    );
}
