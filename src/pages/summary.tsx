import type {ReactNode} from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';

import styles from './summary.module.css';

function SummaryHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero hero--primary', styles.summaryHero)}>
      <div className="container">
        <div className={styles.summaryHeader}>
          <div className={styles.summaryText}>
            <Heading as="h1" className="hero__title">
              Physical AI & Humanoid Robotics
            </Heading>
            <p className="hero__subtitle">A comprehensive guide to building intelligent physical systems</p>
          </div>
          <div className={styles.summaryImage}>
            <img
              src="/img/hero-1.png"
              alt="Physical AI and Humanoid Robotics"
              className={styles.heroImage}
            />
          </div>
        </div>
      </div>
    </header>
  );
}

function ModuleSection({ title, description, image, index }: { title: string; description: string; image: string; index: number }) {
  return (
    <div className={clsx(styles.moduleSection, styles[`module${index}`])}>
      <div className={styles.moduleContent}>
        <div className={styles.moduleImage}>
          <img
            src={image}
            alt={title}
            className={styles.moduleImg}
          />
        </div>
        <div className={styles.moduleText}>
          <Heading as="h3" className={styles.moduleTitle}>{title}</Heading>
          <p className={styles.moduleDescription}>{description}</p>
        </div>
      </div>
    </div>
  );
}

function InteractiveCard({ title, description, icon }: { title: string; description: string; icon: string }) {
  return (
    <div className={styles.interactiveCard}>
      <div className={styles.cardIcon}>{icon}</div>
      <div className={styles.cardContent}>
        <h4>{title}</h4>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function Summary(): ReactNode {
  const {siteConfig} = useDocusaurusContext();

  const modules = [
    {
      title: "Foundations of Physical AI",
      description: "Understanding the core principles of physical intelligence, embodied cognition, and the intersection of AI with physical systems. This module covers the mathematical foundations, sensorimotor integration, and perception-action loops that enable intelligent behavior in physical environments.",
      image: "/img/module-1.png"
    },
    {
      title: "Humanoid Robotics Design",
      description: "Exploring the engineering challenges of building humanoid robots, including mechanical design, actuator systems, balance control, and biomimetic approaches. This module covers kinematics, dynamics, and control systems for creating robots that can navigate and interact with human environments.",
      image: "/img/module-2.png"
    },
    {
      title: "Sensor Integration & Perception",
      description: "Advanced techniques for integrating multiple sensors to enable robots to perceive their environment. This includes computer vision, tactile sensing, proprioception, and multi-modal sensor fusion for robust environmental understanding and interaction.",
      image: "/img/module-3.png"
    },
    {
      title: "Embodied Learning & Adaptation",
      description: "How robots can learn through physical interaction with their environment. This module covers reinforcement learning for physical systems, adaptive control, and the development of motor skills through experience and practice.",
      image: "/img/module-4.png"
    }
  ];

  const keyFeatures = [
    {
      title: "Embodied Intelligence",
      description: "Understanding how intelligence emerges from the interaction between brain, body, and environment",
      icon: "🧠"
    },
    {
      title: "Real-World Application",
      description: "Practical implementations of AI in physical systems with real-world constraints",
      icon: "⚙️"
    },
    {
      title: "Human-Robot Interaction",
      description: "Designing robots that can safely and effectively interact with humans",
      icon: "🤝"
    },
    {
      title: "Adaptive Systems",
      description: "Creating systems that can learn and adapt to new situations and environments",
      icon: "🔄"
    }
  ];

  return (
    <Layout
      title={`Summary - ${siteConfig.title}`}
      description="Physical AI and Humanoid Robotics book summary">
      <SummaryHeader />
      <main className={styles.summaryMain}>
        <section className={styles.introduction}>
          <div className="container">
            <Heading as="h2" className={styles.sectionTitle}>Book Overview</Heading>
            <p className={styles.introductionText}>
              Physical AI and Humanoid Robotics represents a groundbreaking approach to artificial intelligence,
              focusing on the integration of intelligent algorithms with physical systems. This comprehensive guide
              explores how intelligence emerges through the interaction of computational processes with the physical world.
            </p>
            <p className={styles.introductionText}>
              The book addresses the fundamental question of how to create artificial systems that can interact
              with the physical world as effectively as biological systems do. It combines insights from robotics,
              machine learning, neuroscience, and cognitive science to develop a new paradigm for AI development.
            </p>
          </div>
        </section>

        <section className={styles.keyFeatures}>
          <div className="container">
            <Heading as="h2" className={styles.sectionTitle}>Key Features</Heading>
            <div className={styles.cardsContainer}>
              {keyFeatures.map((feature, index) => (
                <InteractiveCard
                  key={index}
                  title={feature.title}
                  description={feature.description}
                  icon={feature.icon}
                />
              ))}
            </div>
          </div>
        </section>

        <section className={styles.modules}>
          <div className="container">
            <Heading as="h2" className={styles.sectionTitle}>Learning Modules</Heading>
            {modules.map((module, index) => (
              <ModuleSection
                key={index}
                title={module.title}
                description={module.description}
                image={module.image}
                index={index + 1}
              />
            ))}
          </div>
        </section>

        <section className={styles.conclusion}>
          <div className="container">
            <div className={styles.conclusionContent}>
              <div className={styles.conclusionText}>
                <Heading as="h2" className={styles.sectionTitle}>Conclusion</Heading>
                <p>
                  Physical AI and Humanoid Robotics represents a paradigm shift in artificial intelligence research,
                  emphasizing the importance of embodiment in creating truly intelligent systems. The book provides
                  both theoretical foundations and practical implementations for developing AI systems that can
                  interact effectively with the physical world.
                </p>
                <p>
                  By understanding the principles of embodied intelligence, readers will be equipped to design
                  and implement the next generation of intelligent physical systems that can learn, adapt, and
                  interact in complex real-world environments.
                </p>
                <div className={styles.ctaButton}>
                  <Link
                    className="button button--primary button--lg"
                    to="/docs">
                    Start Learning Now
                  </Link>
                </div>
              </div>
              <div className={styles.conclusionImage}>
                <img
                  src="/img/hero-2.png"
                  alt="Humanoid Robotics Conclusion"
                  className={styles.heroImage}
                />
              </div>
            </div>
          </div>
        </section>
      </main>
    </Layout>
  );
}