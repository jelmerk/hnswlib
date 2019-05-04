package org.github.jelmerk.foo;

public interface SearchResult<ID, ITEM extends Item<ID>> {

    float getDistance();

    ITEM getItem();
}
