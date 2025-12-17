import type {ReactNode} from 'react';
import clsx from 'clsx';
import Heading from '@theme/Heading';
import styles from './styles.module.css';

type FeatureItem = {
  title: string;
  description: ReactNode;
};

const FeatureList: FeatureItem[] = [
  {
    title: 'Comprehensive Coverage',
    description: (
      <>
        Our AI textbook provides in-depth coverage of artificial intelligence concepts,
        from foundational principles to advanced applications.
      </>
    ),
  },
  {
    title: 'Practical Examples',
    description: (
      <>
        Learn with hands-on examples and real-world applications that demonstrate
        AI concepts in action.
      </>
    ),
  },
  {
    title: 'Up-to-Date Content',
    description: (
      <>
        Stay current with the latest developments in AI technology and research.
      </>
    ),
  },
];

function Feature({title, description}: FeatureItem) {
  return (
    <div className={clsx('col col--4')}>
      <div className="text--center padding-horiz--md">
        <Heading as="h3">{title}</Heading>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures(): ReactNode {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
