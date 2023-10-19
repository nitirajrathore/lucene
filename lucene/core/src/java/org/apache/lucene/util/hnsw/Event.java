package org.apache.lucene.util.hnsw;

/**
 * asdfalksjf lkasjlkfjaslf
 */
public class Event {
    enum Type {
        CONNECT,
        DISCONNECT,
        COMPARE,
        ADD
    }

    public Type type;
    public int source; // primary node of the event
    public int dest;   // secondary node of the event
    public int level; // graph level
    public String otherData;

    public Event(Type type, int source, int dest, int level, String otherData) {
        this.type = type;
        this.source = source;
        this.dest = dest;
        this.level = level;
        this.otherData = otherData;
    }

    @Override
    public String toString() {
        return "Event{" +
                "type=" + type +
                ", source=" + source +
                ", dest=" + dest +
                ", level=" + level +
                ", otherData='" + otherData + '\'' +
                '}';
    }
}
