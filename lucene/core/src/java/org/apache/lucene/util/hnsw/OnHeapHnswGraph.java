/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.lucene.util.hnsw;

import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;

import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.Stack;
import java.util.TreeMap;

import org.apache.lucene.util.Accountable;
import org.apache.lucene.util.RamUsageEstimator;


/**
 * An {@link HnswGraph} where all nodes and connections are held in memory. This class is used to
 * construct the HNSW graph before it's written to the index.
 */
public final class OnHeapHnswGraph extends HnswGraph implements Accountable {
  public Map<Integer, List<Event>> graphEvents = new HashMap<>();

  public void addEvent(int eventKey, Event event) {
    if (!graphEvents.containsKey(eventKey)) {
      graphEvents.put(eventKey, new ArrayList<>());
    }
    graphEvents.get(eventKey).add(event);
  }

  public void clearEvents() {
    graphEvents.clear();
  }

  private int numLevels; // the current number of levels in the graph
  private int entryNode; // the current graph entry node on the top level. -1 if not set

  // Level 0 is represented as List<NeighborArray> – nodes' connections on level 0.
  // Each entry in the list has the top maxConn/maxConn0 neighbors of a node. The nodes correspond
  // to vectors
  // added to HnswBuilder, and the node values are the ordinals of those vectors.
  // Thus, on all levels, neighbors expressed as the level 0's nodes' ordinals.
  private final List<NeighborArray> graphLevel0;
  // Represents levels 1-N. Each level is represented with a TreeMap that maps a levels level 0
  // ordinal to its neighbors on that level. All nodes are in level 0, so we do not need to maintain
  // it in this list. However, to avoid changing list indexing, we always will make the first
  // element
  // null.
  private final List<TreeMap<Integer, NeighborArray>> graphUpperLevels;
  private final int nsize;
  private final int nsize0;

  // KnnGraphValues iterator members
  private int upto;
  private NeighborArray cur;

  OnHeapHnswGraph(int M) {
    this.numLevels = 1; // Implicitly start the graph with a single level
    this.graphLevel0 = new ArrayList<>();
    this.entryNode = -1; // Entry node should be negative until a node is added
    // Neighbours' size on upper levels (nsize) and level 0 (nsize0)
    // We allocate extra space for neighbours, but then prune them to keep allowed maximum
    this.nsize = M + 1;
    this.nsize0 = (M * 2 + 1);

    this.graphUpperLevels = new ArrayList<>(numLevels);
    graphUpperLevels.add(null); // we don't need this for 0th level, as it contains all nodes
  }

  /**
   * Returns the {@link NeighborQueue} connected to the given node.
   *
   * @param level level of the graph
   * @param node the node whose neighbors are returned, represented as an ordinal on the level 0.
   */
  public NeighborArray getNeighbors(int level, int node) {
    if (level == 0) {
      return graphLevel0.get(node);
    }
    TreeMap<Integer, NeighborArray> levelMap = graphUpperLevels.get(level);
    assert levelMap.containsKey(node);
    return levelMap.get(node);
  }

  @Override
  public int size() {
    return graphLevel0.size(); // all nodes are located on the 0th level
  }

  /**
   * Add node on the given level. Nodes can be inserted out of order, but it requires that the nodes
   * preceded by the node inserted out of order are eventually added.
   *
   * @param level level to add a node on
   * @param node the node to add, represented as an ordinal on the level 0.
   */
  public void addNode(int level, int node) {
    addEvent(node, new Event(Event.Type.ADD, node, -1, level, ""));
    if (entryNode == -1) {
      entryNode = node;
    }

    if (level > 0) {
      // if the new node introduces a new level, add more levels to the graph,
      // and make this node the graph's new entry point
      if (level >= numLevels) {
        for (int i = numLevels; i <= level; i++) {
          graphUpperLevels.add(new TreeMap<>());
        }
        numLevels = level + 1;
        entryNode = node;
      }

      graphUpperLevels.get(level).put(node, new NeighborArray(nsize, true));
    } else {
      // Add nodes all the way up to and including "node" in the new graph on level 0. This will
      // cause the size of the
      // graph to differ from the number of nodes added to the graph. The size of the graph and the
      // number of nodes
      // added will only be in sync once all nodes from 0...last_node are added into the graph.
      while (node >= graphLevel0.size()) {
        graphLevel0.add(new NeighborArray(nsize0, true));
      }
    }
  }

  @Override
  public void seek(int level, int targetNode) {
    cur = getNeighbors(level, targetNode);
    upto = -1;
  }

  @Override
  public int nextNeighbor() {
    if (++upto < cur.size()) {
      return cur.node[upto];
    }
    return NO_MORE_DOCS;
  }

  /**
   * Returns the current number of levels in the graph
   *
   * @return the current number of levels in the graph
   */
  @Override
  public int numLevels() {
    return numLevels;
  }

  /**
   * Returns the graph's current entry node on the top level shown as ordinals of the nodes on 0th
   * level
   *
   * @return the graph's current entry node on the top level
   */
  @Override
  public int entryNode() {
    return entryNode;
  }

  @Override
  public NodesIterator getNodesOnLevel(int level) {
    if (level == 0) {
      // Since the zeroth level graph has nodes from 0 to size()-1 elements ALWAYS, there is no need to keep these
      // values in
      // nodes array, just the size is enough. The nodes == null cases is handled in all internal methods of
      // ArrayNodesIterator
      return new ArrayNodesIterator(size());
    } else {
      return new CollectionNodesIterator(graphUpperLevels.get(level).keySet());
    }
  }

  public static Set<Integer> findReacheable(OnHeapHnswGraph graph, int level) {
    Set<Integer> visited = new HashSet<>();
    Stack<Integer> candidates = new Stack<>();
    candidates.push(graph.entryNode());

    while (!candidates.isEmpty()) {
      int node = candidates.pop();

      if (visited.contains(node)) {
        continue;
      }

      visited.add(node);
      graph.seek(level, node);

      int friendOrd;
      while ((friendOrd = graph.nextNeighbor()) != NO_MORE_DOCS) {
        candidates.push(friendOrd);
      }
    }
    return visited;
  }
  @Override
  public long ramBytesUsed() {
    long neighborArrayBytes0 =
        nsize0 * (Integer.BYTES + Float.BYTES)
            + RamUsageEstimator.NUM_BYTES_ARRAY_HEADER
            + RamUsageEstimator.NUM_BYTES_OBJECT_REF * 2
            + Integer.BYTES * 3;
    long neighborArrayBytes =
        nsize * (Integer.BYTES + Float.BYTES)
            + RamUsageEstimator.NUM_BYTES_ARRAY_HEADER
            + RamUsageEstimator.NUM_BYTES_OBJECT_REF * 2
            + Integer.BYTES * 3;
    long total = 0;
    for (int l = 0; l < numLevels; l++) {
      if (l == 0) {
        total +=
            graphLevel0.size() * neighborArrayBytes0
                + RamUsageEstimator.NUM_BYTES_OBJECT_REF; // for graph;
      } else {
        long numNodesOnLevel = graphUpperLevels.get(l).size();

        // For levels > 0, we represent the graph structure with a tree map.
        // A single node in the tree contains 3 references (left root, right root, value) as well
        // as an Integer for the key and 1 extra byte for the color of the node (this is actually 1
        // bit, but
        // because we do not have that granularity, we set to 1 byte). In addition, we include 1
        // more reference for
        // the tree map itself.
        total +=
            numNodesOnLevel * (3L * RamUsageEstimator.NUM_BYTES_OBJECT_REF + Integer.BYTES + 1)
                + RamUsageEstimator.NUM_BYTES_OBJECT_REF;

        // Add the size neighbor of each node
        total += numNodesOnLevel * neighborArrayBytes;
      }
    }
    return total;
  }

  public void printConnections(FileWriter fw) throws IOException {
    int node = 0;
    for (NeighborArray na : graphLevel0) {
      fw.write(getNeighbourString(0, node, na.size()) + "\n");
      node++;
    }
    for (int level = 1; level < numLevels; level++) {
      int finalLevel = level;
      graphUpperLevels.get(level).forEach((key, value) -> {
        try {
          fw.write(getNeighbourString(finalLevel, key, value.size()) + "\n");
        } catch (IOException e) {
          throw new RuntimeException(e);
        }
      });
    }
  }

  private String getNeighbourString(int level, int node, int count) {
    return "level=" + level + ",node=" + node + ",count=" + count;
  }

  public Set<Integer> findUnReacheable(int level) {
    Set<Integer> reachable = findReacheable(this, level);
    if (level == 0) {
      Set<Integer> unrechableNodes = new HashSet<>();
      for (int i = 0; i < this.size(); i++) {
        if (!reachable.contains(i)) {
          unrechableNodes.add(i);
        }
      }
      return unrechableNodes;
    }
    Set<Integer> nodes = new HashSet<Integer>(this.graphUpperLevels.get(level).keySet()); // total nodes
    nodes.removeAll(reachable);
    return nodes;
  }

  private static Set<Integer> getOverallReachableNodes(HnswGraph graph, int numLevels) throws IOException {
    Set<Integer> visited = new HashSet<>();
    Stack<Integer> candidates = new Stack<>();
    candidates.push(graph.entryNode());

    while (!candidates.isEmpty()) {
      int node = candidates.pop();

      if (visited.contains(node)) {
        continue;
      }

      visited.add(node);
      // emulate collapsing all the edges
      for (int level = 0; level < numLevels; level++) {
        try {
          graph.seek(level, node); // run with -ea (Assertions enabled)
          int friendOrd;
          while ((friendOrd = graph.nextNeighbor()) != NO_MORE_DOCS) {
            candidates.push(friendOrd);
          }
        } catch (AssertionError ae) { // TODO: seriously? Bad but have to do it for this test, as the API does not
// throw error. You will get this error when the graph.seek() tries to seek a node which is not present in that level.
        }
      }
    }
    return visited;
  }

  public List<Integer> getOverAllDisconnectedNodes() throws IOException {
    Set<Integer> reachable = getOverallReachableNodes(this, numLevels);
    List<Integer> unrechableNodes = new ArrayList<>();
    for (int i = 0; i < this.size(); i++) {
      if (!reachable.contains(i)) {
        unrechableNodes.add(i);
      }
    }
    return unrechableNodes;
  }
}
