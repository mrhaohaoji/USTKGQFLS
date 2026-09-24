**Algorithm 1**: Uncertain Spatiotemporal Knowledge Graph (USTKG) Dataset Construction

**Input**: Wikidata, YAGO

**Output**: USTKG dataset (*train*, *valid*, *test*)

1\. **Initialize** empty set *S*.

2\. Extract from Wikidata and YAGO:

&#x20;     relation triples *R* = {(*h*, *r*, *t*)}

&#x20;     temporal annotations *T*: entity *e* -> set of timestamps or intervals

&#x20;     spatial annotations *L*: entity *e* -> set of (latitude, longitude)

3\. **For** each triple (*h*, *r*, *t*) in *R*:

&#x20;     Get timestamps for *h* and t: *Th* = *T*.*get*(*h*, *empty*), *Tt* = *T*.*get*(*t*, *empty*)

&#x20;     Get locations for *h* and *t*: *Lh* = *L*.*get*(*h*, *empty*), *Lt* = *L*.*get*(*t*, *empty*)

&#x20;     **If** *Th* is empty and *Tt* is empty: skip (missing temporal info)

&#x20;     **If** *Lh* is empty and *Lt* is empty: skip (missing spatial info)

&#x20;     **For** each timestamp *τ* in (*Th* ∪ *Tt*):

&#x20;       **For** each location *l* in (*Lh* ∪ *Lt*):

&#x20;           **If** *τ* is an interval \[*τstart*, *τend*]:

&#x20;               Split into τstart and τend, r -> r\_ start and r\_end, add two tuples.

&#x20;           **Else**:

&#x20;               Add (*h*, *r*, *t*, *τ*, *l*) to *S*.

4\. **For** each tuple (*h*, *r*, *t*, *τ*, *l*) in *S*:

&#x20;     Generate membership *μ* -> *Triangular*(*a*=0, *b*=0.9, *c*=1)

&#x20;     Form sextuple (*h*, *r*, *t*, *τ*, *l*, *μ*)

&#x20;     Add to final dataset *D*.

5\. Remove duplicate sextuples from *D*.

6\. Randomly split *D* into *train*/*valid*/*test* = 8:1:1.

7\. **Return** *train*, *valid*, *test*.

