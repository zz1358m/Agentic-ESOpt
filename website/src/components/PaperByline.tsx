import { Fragment } from "react";
import { paper } from "../paper";

export function PaperByline() {
  return (
    <div className="paper-byline" role="group" aria-label="Paper authors and affiliations">
      <p className="paper-byline__authors">
        {paper.authors.map((author, index) => (
          <Fragment key={author.name}>
            {index > 0 ? " · " : null}
            <span>{author.name}<sup>{author.affiliation}</sup></span>
          </Fragment>
        ))}
      </p>
      <ul>
        {paper.affiliations.map((affiliation) => (
          <li key={affiliation.id} aria-label={`${affiliation.id} ${affiliation.name}`}><sup>{affiliation.id}</sup> {affiliation.name}</li>
        ))}
      </ul>
      <p className="paper-byline__correspondence">
        Correspondence <a href={`mailto:${paper.correspondingEmail}`}>{paper.correspondingEmail}</a>
      </p>
    </div>
  );
}
